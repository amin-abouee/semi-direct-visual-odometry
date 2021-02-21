#include "system.hpp"
#include "algorithm.hpp"
#include "utils.hpp"
#include "visualization.hpp"

#include <easylogging++.h>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>

#include <chrono>
#include <iostream>

#define System_Log( LEVEL ) CLOG( LEVEL, "System" )

System::System( const std::shared_ptr< Config >& config )
    : m_config( config )
    , m_activeKeyframe( nullptr )
    , predictionRelativePose( Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero() )
    , m_systemStatus( System::Status::Process_First_Frame )
{
    const std::string calibrationFile = utils::findAbsoluteFilePath( m_config->m_cameraCalibrationPath );
    cv::Mat cameraMatrix;
    cv::Mat distortionCoeffs;
    loadCameraIntrinsics( calibrationFile, cameraMatrix, distortionCoeffs );
    m_camera          = std::make_shared< PinholeCamera >( m_config->m_imgWidth, m_config->m_imgHeight, cameraMatrix, distortionCoeffs );
    m_alignment       = std::make_shared< ImageAlignment >( m_config->m_patchSizeImageAlignment, m_config->m_minLevelImagePyramid,
                                                      m_config->m_maxLevelImagePyramid, 6 );
    m_featureSelector = std::make_shared< FeatureSelection >( m_config->m_imgWidth, m_config->m_imgHeight, m_config->m_cellPixelSize );
    m_map             = std::make_shared< Map >( m_camera, m_config->m_cellPixelSize );
    m_depthEstimator  = std::make_unique< DepthEstimator >( m_map, m_featureSelector );
    m_bundler         = std::make_shared< BundleAdjustment >( m_camera, 0, 6 );
}

bool System::addImage( const cv::Mat& img, const uint64_t timestamp )
{
    m_curFrame = std::make_shared< Frame >( m_camera, img, m_config->m_maxLevelImagePyramid + 1, timestamp, m_activeKeyframe );
    // m_allFrames.emplace_back( m_curFrame );
    System_Log( INFO ) << "Processing frame id: " << m_curFrame->m_id;
    System::Result res = System::Result::Failed;

    if ( m_systemStatus == System::Status::Procese_New_Frame )
    {
        res = processNewFrame();
    }
    else if ( m_systemStatus == System::Status::Process_Second_Frame )
    {
        // TODO: check to avoid change of reference when res for the second frame is false
        res = processSecondFrame();
    }
    else if ( m_systemStatus == System::Status::Process_First_Frame )
    {
        res = processFirstFrame();
    }
    else if ( m_systemStatus == System::Status::Process_Relocalization )
    {
        std::shared_ptr< Frame > closestKeyframe{ nullptr };
        m_map->getClosestKeyframe( m_curFrame, closestKeyframe );
        Sophus::SE3d pose;
        res = relocalizeFrame( pose, closestKeyframe );
    }

    if ( m_refFrame != nullptr )
    {
        predictionRelativePose = algorithm::computeRelativePose( m_refFrame, m_curFrame );
    }

    if ( res == System::Result::Failed )
    {
        return false;
    }
    else
    {
        m_refFrame = std::move( m_curFrame );
        return true;
    }
}

System::Result System::processFirstFrame()
{
    TIMED_FUNC( timerFirstFrame );
    m_featureSelector->gradientMagnitudeWithSSC( m_curFrame, m_config->m_thresholdGradientMagnitude,
                                                 m_config->m_desiredDetectedPointsForInitialization, true );
    // m_featureSelector->gradientMagnitudeByValue(m_curFrame, m_config->m_thresholdGradientMagnitude, true);
    if ( m_curFrame->numberObservation() < m_config->m_minDetectedPointsSuccessInitialization )
    {
        System_Log( WARNING ) << "Not sufficient detected feature points!";
        return Result::Failed;
    }

    // visualize
    if ( m_config->m_enableVisualization == true )
    {
        cv::Mat gradient = m_featureSelector->m_imgGradientMagnitude.clone();
        cv::Mat refBGR   = visualization::getColorImage( gradient );
        visualization::featurePoints( refBGR, m_curFrame, 5, "pink", true, visualization::drawingRectangle );
        visualization::imageGrid( refBGR, m_config->m_cellPixelSize, "amber" );
        if ( m_config->m_savingType == "LiveShow" )
        {
            cv::imshow( "First Image", refBGR );
        }
        else if ( m_config->m_savingType == "File" )
        {
            std::stringstream ss;
            ss << "../output/images/" << m_curFrame->m_id << "_" << m_curFrame->m_id << ".jpg";
            cv::imwrite( ss.str(), refBGR );
        }
    }

    m_curFrame->setKeyframe();
    m_activeKeyframe = m_curFrame;
    m_map->addKeyframe( m_curFrame );
    System_Log( DEBUG ) << "Number of Features: " << m_curFrame->numberObservation();
    m_systemStatus = System::Status::Process_Second_Frame;
    reportSummary();
    return Result::Success;
}

System::Result System::processSecondFrame()
{
    TIMED_FUNC( timerSecondFrame );

    // TODO: check current id and ref id. if is more than n frame, reset ref frame

    Eigen::Matrix3d E;
    Eigen::Matrix3d R;
    Eigen::Vector3d t;

    // we find the corresponding point in curFrame and create feature for them
    bool resOpticalFlow =
      algorithm::computeOpticalFlowSparse( m_refFrame, m_curFrame, m_config->m_patchSizeOpticalFlow, m_config->m_disparityThreshold );
    if ( resOpticalFlow == false )
    {
        System_Log( WARNING ) << "Disparity (displacement) is not sufficient";
        return Result::Failed;
    }

    const uint32_t thresholdCorrespondingPoints = m_config->m_minDetectedPointsSuccessInitialization * 0.5;
    bool resEssentialMatrix = algorithm::computeEssentialMatrix( m_refFrame, m_curFrame, 1.0, thresholdCorrespondingPoints, E );
    if ( resEssentialMatrix == false )
    {
        System_Log( WARNING ) << "Number of corresponding Points between ref and current is less than you threshold";
        return Result::Failed;
    }

    const Eigen::Matrix3d F = m_refFrame->m_camera->invK().transpose() * E * m_curFrame->m_camera->invK();
    algorithm::sampsonCorrection( m_refFrame, m_curFrame, F );

    // decompose essential matrix to R and t and set the absolute pose of reference frame
    bool resRecoverPose = algorithm::recoverPose( E, m_refFrame, m_curFrame, R, t );
    if ( resRecoverPose == false )
    {
        System_Log( WARNING ) << "Pose recovering failed";
        return Result::Failed;
    }

    System_Log( DEBUG ) << "Initial R: " << R.format( utils::eigenFormat() );
    System_Log( DEBUG ) << "Initial t: " << t.format( utils::eigenFormat() );

    // reserve the memory for depth information
    std::size_t numObserves = m_curFrame->numberObservation();
    Eigen::MatrixXd pointsWorld( 3, numObserves );
    Eigen::MatrixXd pointsCurCamera( 3, numObserves );
    Eigen::VectorXd depthCurFrame( numObserves );

    // compute the 3D points
    algorithm::triangulate3DWorldPoints( m_refFrame, m_curFrame, pointsWorld );
    algorithm::transferPointsWorldToCam( m_curFrame, pointsWorld, pointsCurCamera );
    algorithm::normalizedDepthCamera( m_curFrame, pointsWorld, depthCurFrame );
    double medianDepth = algorithm::computeMedian( depthCurFrame );
    {
        const double minDepth = depthCurFrame.minCoeff();
        const double maxDepth = depthCurFrame.maxCoeff();
        System_Log( INFO ) << "Before SCALE, Median Depth: " << medianDepth << ", minDepth: " << minDepth << ", maxDepth: " << maxDepth;
    }

    // scale the depth
    const double scale = m_config->m_initMapScaleFactor / medianDepth;

    // t = -RC
    // In this formula, we assume, the W = K_{1}, means the m_refFrame->cameraInWorld() = (0, 0, 0)
    // first we compute the C, scale it and compute t
    m_curFrame->m_absPose.translation() =
      -m_curFrame->m_absPose.rotationMatrix() *
      ( m_refFrame->cameraInWorld() + scale * ( m_curFrame->cameraInWorld() - m_refFrame->cameraInWorld() ) );

    // scale current frame depth
    pointsCurCamera *= scale;
    algorithm::transferPointsCamToWorld( m_curFrame, pointsCurCamera, pointsWorld );

    uint32_t cnt = 0;
    for ( std::size_t i( 0 ); i < numObserves; i++ )
    {
        const Eigen::Vector2d refFeature = m_refFrame->m_features[ i ]->m_pixelPosition;
        const Eigen::Vector2d curFeature = m_curFrame->m_features[ i ]->m_pixelPosition;
        if ( m_refFrame->m_camera->isInFrame( refFeature, 5.0 ) == true && m_curFrame->m_camera->isInFrame( curFeature, 5.0 ) == true &&
             pointsCurCamera.col( i ).z() > 0 )
        {
            std::shared_ptr< Point > point = std::make_shared< Point >( pointsWorld.col( i ) );
            m_refFrame->m_features[ i ]->setPoint( point );
            m_curFrame->m_features[ i ]->setPoint( point );
            point->addFeature( m_refFrame->m_features[ i ] );
            point->addFeature( m_curFrame->m_features[ i ] );
            cnt++;
        }
    }

    // remove the features that doesn't have the 3D point
    m_refFrame->m_features.erase( std::remove_if( m_refFrame->m_features.begin(), m_refFrame->m_features.end(),
                                                  []( const auto& feature ) { return feature->m_point == nullptr; } ),
                                  m_refFrame->m_features.end() );

    m_curFrame->m_features.erase( std::remove_if( m_curFrame->m_features.begin(), m_curFrame->m_features.end(),
                                                  []( const auto& feature ) { return feature->m_point == nullptr; } ),
                                  m_curFrame->m_features.end() );

    System_Log( INFO ) << "Init Points: " << pointsWorld.cols() << ", ref obs: " << m_refFrame->numberObservation()
                       << ", cur obs: " << m_curFrame->numberObservation() << ", number of removed: " << pointsWorld.cols() - cnt;

    {
        TIMED_SCOPE( timerBA, "timer twoViewBA" );
        m_bundler->twoViewBA( m_refFrame, m_curFrame, m_map, 2.0 );
    }

    m_curFrame->setKeyframe();
    m_activeKeyframe = m_curFrame;
    numObserves      = m_curFrame->numberObservation();
    Eigen::VectorXd newCurDepths( numObserves );
    algorithm::normalizedDepthCamera( m_curFrame, newCurDepths );
    medianDepth           = algorithm::computeMedian( newCurDepths );
    const double minDepth = newCurDepths.minCoeff();
    const double maxDepth = newCurDepths.maxCoeff();
    System_Log( INFO ) << "After scale, Median depth: " << medianDepth << ", minDepth: " << minDepth << ", maxDepth: " << maxDepth;

    System_Log( INFO ) << "Size observation: " << m_curFrame->numberObservation();
    m_featureSelector->setExistingFeatures( m_curFrame->m_features );
    m_featureSelector->gradientMagnitudeWithSSC( m_curFrame, m_config->m_thresholdGradientMagnitude,
                                                 m_config->m_desiredDetectedPointsForInitialization, true );
    // m_featureSelector->gradientMagnitudeByValue(m_curFrame, m_config->m_thresholdGradientMagnitude, true);

    System_Log( INFO ) << "Size observation after detect: " << m_curFrame->numberObservation();

    m_depthEstimator->addKeyframe( m_curFrame, medianDepth, 0.5 * minDepth );

    m_map->addKeyframe( m_curFrame );

    if ( m_config->m_enableVisualization == true )
    {
        cv::Mat refBGR = visualization::getColorImage( m_refFrame->m_imagePyramid.getBaseGradientImage() );
        visualization::featurePoints( refBGR, m_refFrame, 6, "pink", true, visualization::drawingRectangle );

        cv::Mat curBGR = visualization::getColorImage( m_curFrame->m_imagePyramid.getBaseGradientImage() );
        visualization::featurePoints( curBGR, m_curFrame, 6, "orange", false, visualization::drawingCircle );
        // visualization::projectPointsWithRelativePose( curBGR, m_refFrame, m_curFrame, 6, "orange", visualization::drawingCircle );

        cv::Mat stickImg;
        visualization::stickTwoImageVertically( refBGR, curBGR, stickImg );

        // cv::Mat curBGR2 = visualization::getColorImage( m_curFrame->m_imagePyramid.getBaseGradientImage() );
        // visualization::imageGrid( curBGR2, m_config->m_cellPixelSize, "green" );

        // visualization::stickTwoImageVertically( stickImg, curBGR2, stickImg );

        // cv::Mat curBGR3 = visualization::getColorImage( m_curFrame->m_imagePyramid.getBaseGradientImage() );
        // visualization::project3DPoints( curBGR3, m_curFrame, 6, "red", visualization::drawingCircle );

        // visualization::stickTwoImageVertically( stickImg, curBGR3, stickImg );

        if ( m_config->m_savingType == "LiveShow" )
        {
            std::stringstream ss;
            ss << m_refFrame->m_id << "_" << m_curFrame->m_id;
            cv::imshow( ss.str(), stickImg );
        }
        else if ( m_config->m_savingType == "File" )
        {
            std::stringstream ss;
            ss << "../output/images/" << m_refFrame->m_id << "_" << m_curFrame->m_id << ".jpg";
            cv::imwrite( ss.str(), stickImg );
        }

        // visualization::projectLinesWithRelativePose( curBGR, m_refFrame, m_curFrame, 12, "cyan", "indigo", visualization::drawingLine );

        // if ( m_config->m_savingType == "LiveShow" )
        // {
        //     std::stringstream ss;
        //     ss << m_refFrame->m_id << " -> " << m_curFrame->m_id;
        //     cv::imshow( ss.str(), stickImg );
        // }
        // else if ( m_config->m_savingType == "File" )
        // {
        //     std::stringstream ss;
        //     ss << "../output/images/epipolar.png";
        //     cv::imwrite( ss.str(), curBGR );
        // }

        // cv::Mat curBGR2 = visualization::getColorImage( m_curFrame->m_imagePyramid.getBaseGradientImage() );
        // visualization::featurePoints( curBGR2, m_curFrame, 6, "orange", true, visualization::drawingCircle );

        // if ( m_config->m_savingType == "LiveShow" )
        // {
        //     std::stringstream ss;
        //     ss << m_refFrame->m_id << " -> " << m_curFrame->m_id;
        //     cv::imshow( ss.str(), stickImg );
        // }
        // else if ( m_config->m_savingType == "File" )
        // {
        //     std::stringstream ss;
        //     ss << "../output/images/new_features.png";
        //     cv::imwrite( ss.str(), curBGR2 );
        // }

        // cv::Mat curBGR3 = visualization::getColorImage( m_curFrame->m_imagePyramid.getBaseGradientImage() );
        // visualization::colormapDepth(curBGR3, m_curFrame, 6);

        // if ( m_config->m_savingType == "LiveShow" )
        // {
        //     std::stringstream ss;
        //     ss << m_refFrame->m_id << " -> " << m_curFrame->m_id;
        //     cv::imshow( ss.str(), stickImg );
        // }
        // else if ( m_config->m_savingType == "File" )
        // {
        //     std::stringstream ss;
        //     ss << "../output/images/depth.png";
        //     cv::imwrite( ss.str(), curBGR3 );
        // }
    }

    m_systemStatus = System::Status::Procese_New_Frame;
    reportSummary();
    return Result::Success;
}

System::Result System::processNewFrame()
{
    TIMED_FUNC( timerNewFrame );
    // https://docs.microsoft.com/en-us/cpp/cpp/how-to-create-and-use-shared-ptr-instances?view=vs-2019

    m_curFrame->m_absPose = predictionRelativePose * m_refFrame->m_absPose;
    {
        TIMED_SCOPE( timerImageAlignment, "image_alignment" );
        m_alignment->align( m_refFrame, m_curFrame );
    }

    m_map->addCandidateToAllActiveKeyframes();

    cv::Mat stickImg;
    if ( m_config->m_enableVisualization == true )
    {
        cv::Mat lastBGR = visualization::getColorImage( m_refFrame->m_lastKeyframe->m_imagePyramid.getBaseGradientImage() );
        visualization::featurePoints( lastBGR, m_refFrame->m_lastKeyframe, 6, "green", true,
                                      visualization::drawingRectangle );

        cv::Mat refBGR = visualization::getColorImage( m_refFrame->m_imagePyramid.getBaseGradientImage() );
        visualization::featurePoints( refBGR, m_refFrame, 6, "pink", true, visualization::drawingRectangle );

        visualization::stickTwoImageVertically( lastBGR, refBGR, stickImg, 20 );

        // just projected with computed pose
        // cv::Mat curBGR = visualization::getColorImage( m_curFrame->m_imagePyramid.getBaseGradientImage() );
        // visualization::projectPointsWithRelativePose( curBGR, m_refFrame, m_curFrame, 6, "yellow", visualization::drawingCircle );

        // visualization::stickTwoImageVertically( stickImg, curBGR, stickImg, 20 );
    }

    std::vector< frameSize > overlapKeyFrames;
    m_map->reprojectMap( m_refFrame, m_curFrame, overlapKeyFrames );
    System_Log( INFO ) << "Number of Features in new frame: " << m_curFrame->numberObservation();
    double befError = algorithm::computeStructureError(m_curFrame);

/*
    {
        std::vector< cv::Point3d > objectPoints;
        std::vector< cv::Point2d > imagePoints;

        const Eigen::Vector2d focalLength = m_curFrame->m_camera->focalLength();
        const Eigen::Vector2d principlePoint = m_curFrame->m_camera->principalPoint();
        cv::Matx33d K (focalLength.x(), 0.0, principlePoint.x(), 0.0, focalLength.y(), principlePoint.y(), 0.0, 0.0, 1.0);
        cv::Mat dist_coeffs = cv::Mat::zeros(4,1,cv::DataType<double>::type);

        for ( const auto& feature : m_curFrame->m_features )
        {
            if ( feature->m_point != nullptr )
            {
                const auto& point = feature->m_point;
                imagePoints.push_back( cv::Point2d( feature->m_pixelPosition.x(), feature->m_pixelPosition.y() ) );
                objectPoints.push_back( cv::Point3d( point->m_position.x(), point->m_position.y(), point->m_position.z() ) );
            }
        }

        const Eigen::Matrix3d computedRot = m_curFrame->m_absPose.rotationMatrix();
        const Eigen::Vector3d computedTranslation = m_curFrame->m_absPose.translation();
        System_Log( WARNING ) << "init trans: " << m_curFrame->m_absPose.params().transpose();


        cv::Mat cv_R = (cv::Mat_<double>(3,3) << computedRot(0, 0), computedRot(0, 1), computedRot(0, 2), computedRot(1, 0), computedRot(1, 1), computedRot(1, 2), computedRot(2, 0), computedRot(2, 1), computedRot(2, 2));
        cv::Mat rotation_vector;
        cv::Rodrigues(cv_R, rotation_vector);
        cv::Mat translation_vector = (cv::Mat_<double>(3,1) << computedTranslation (0), computedTranslation(1), computedTranslation(2));

        // System_Log( WARNING ) << "eigen rot: " << computedRot;
        // System_Log( WARNING ) << "rotation: " << cv_R;

        cv::solvePnP(objectPoints, imagePoints, K, dist_coeffs, rotation_vector, translation_vector, false);

        m_curFrame->m_absPose.translation() = Eigen::Vector3d (translation_vector.at<double>(0,0), translation_vector.at<double>(1,0), translation_vector.at<double>(2,0));

        cv::Mat outputRotation;
        cv::Rodrigues(rotation_vector, outputRotation);

        Eigen::Matrix3d finalRotation;
        finalRotation << outputRotation.at<double>(0,0), outputRotation.at<double>(0,1), outputRotation.at<double>(0,2), outputRotation.at<double>(1,0), outputRotation.at<double>(1,1), outputRotation.at<double>(1,2), outputRotation.at<double>(2,0), outputRotation.at<double>(2,1), outputRotation.at<double>(2,2);

        // System_Log( WARNING ) << "rotation: " << outputRotation;
        // System_Log( WARNING ) << "eigen rot: " << finalRotation;

        Eigen::AngleAxisd tmpRot(finalRotation);
        m_curFrame->m_absPose.setRotationMatrix(tmpRot.toRotationMatrix());

        System_Log( WARNING ) << "final trans: " << m_curFrame->m_absPose.params().transpose();
        // System_Log( INFO ) << "rotation new:" << m_curFrame->m_absPose.rotationMatrix();
    }
*/

    // m_bundler->optimizePose(m_curFrame);
    // m_bundler->oneFrameWithScene (m_curFrame, 2.0);
    m_bundler->optimizeScene(m_curFrame, 2.0);
    // m_bundler->threeViewBA(m_curFrame, 2.0);
    double aftError = algorithm::computeStructureError(m_curFrame);
    System_Log( INFO ) << "Structure error: " << befError << " -> " << aftError;

    if ( m_config->m_enableVisualization == true )
    {
        // points projected and added
        cv::Mat curBGR = visualization::getColorImage( m_curFrame->m_imagePyramid.getBaseGradientImage() );
        visualization::featurePoints( curBGR, m_curFrame, 6, "orange", false, visualization::drawingCircle );

        visualization::stickTwoImageVertically( stickImg, curBGR, stickImg, 20 );

        cv::Mat curBGR2 = visualization::getColorImage( m_curFrame->m_imagePyramid.getBaseGradientImage() );
        visualization::featurePointsAndProjection( curBGR2, m_curFrame, 6, "orange", "yellow", visualization::drawingCircle );

        visualization::stickTwoImageVertically( stickImg, curBGR2, stickImg, 20 );


        if ( m_config->m_savingType == "LiveShow" )
        {
            std::stringstream ss;
            ss << m_refFrame->m_id << " -> " << m_curFrame->m_id;
            cv::imshow( ss.str(), stickImg );
        }
        else if ( m_config->m_savingType == "File" )
        {
            std::stringstream ss;
            ss << "../output/images/" << m_refFrame->m_id << " -> " << m_curFrame->m_id << ".png";
            cv::imwrite( ss.str(), stickImg );
        }
    }

    // System_Log (DEBUG) << "m_curFrame->m_lastKeyframe id: " << m_curFrame->m_lastKeyframe->m_id;
    // m_map->addCandidateToFrame( m_curFrame->m_lastKeyframe );

    // we don't add features, if visited frame is key frame, because it can change or distribution

    if ( m_config->m_enableVisualization == true )
    {
        // cv::Mat lastBGR = visualization::getColorImage( m_refFrame->m_lastKeyframe->m_imagePyramid.getBaseGradientImage() );
        // visualization::featurePoints( lastBGR, m_refFrame->m_lastKeyframe, 6, "green", true, visualization::drawingRectangle );

        // visualization::stickTwoImageVertically( stickImg, lastBGR, stickImg, 20 );

        // cv::Mat refBGR = visualization::getColorImage( m_refFrame->m_imagePyramid.getBaseGradientImage() );
        // visualization::featurePoints( refBGR, m_refFrame, 6, "pink", true, visualization::drawingRectangle );

        // // cv::Mat stickImg;
        // visualization::stickTwoImageVertically( stickImg, refBGR, stickImg, 20 );

        // if ( m_config->m_savingType == "LiveShow" )
        // {
        //     std::stringstream ss;
        //     ss << m_refFrame->m_id << " -> " << m_curFrame->m_id;
        //     cv::imshow( ss.str(), stickImg );
        // }
        // else if ( m_config->m_savingType == "File" )
        // {
        //     std::stringstream ss;
        //     ss << "../output/images/" << m_refFrame->m_id << " -> " << m_curFrame->m_id << ".png";
        //     cv::imwrite( ss.str(), stickImg );
        // }
    }

    // m_bundler->optimizePose( m_curFrame );
    // m_bundler->optimizeStructure( m_curFrame, 50 );
    // uint32_t obsWithPoints = 0;
    // for ( const auto& feature : m_refFrame->m_features )
    // {
    //     if ( feature->m_point != nullptr )
    //     {
    //         obsWithPoints++;
    //     }
    // }
    uint32_t obsWithPoints = m_refFrame->numberObservationWithPoints();
    bool qualityCheck      = computeTrackingQuality( m_curFrame, obsWithPoints );
    if ( qualityCheck == false )
    {
        m_curFrame->m_absPose = m_refFrame->m_absPose;
        return Result::Success;
    }

    const uint32_t numObserves = m_curFrame->numberObservation();
    Eigen::VectorXd depthsInCurFrame( numObserves );
    algorithm::normalizedDepthCamera( m_curFrame, depthsInCurFrame );
    const double depthMean = algorithm::computeMedian( depthsInCurFrame );
    const double depthMin  = depthsInCurFrame.minCoeff();

    if ( needKeyframe( depthsInCurFrame, depthMean ) )
    {
        m_depthEstimator->addFrame( m_curFrame );
        reportSummary();
        return Result::Success;
    }

    m_curFrame->setKeyframe();
    m_map->addKeyframe( m_curFrame );
    m_activeKeyframe = m_curFrame;

    {
        TIMED_SCOPE( timerBA, "timer local BA" );
        m_bundler->localBA( m_map, 2.0 );
    }

    m_featureSelector->setExistingFeatures( m_curFrame->m_features );
    m_featureSelector->gradientMagnitudeWithSSC( m_curFrame, m_config->m_thresholdGradientMagnitude,
                                                 m_config->m_desiredDetectedPointsForInitialization, true );

    // TODO: run feature selection for missing part
    // run depth estimation for new points
    m_depthEstimator->addKeyframe( m_curFrame, depthMean, depthMin * 0.5 );

    // remove old key frame from map
    if ( m_map->m_keyFrames.size() > 5 )
    {
        std::shared_ptr< Frame > furthestFrame{ nullptr };
        m_map->getFurthestKeyframe( m_curFrame->cameraInWorld(), furthestFrame );
        m_depthEstimator->removeKeyframe( furthestFrame );
        m_map->removeFrame( furthestFrame );
    }

    reportSummary();
    return Result::Keyframe;
}

System::Result System::relocalizeFrame( Sophus::SE3d& pose, std::shared_ptr< Frame >& closestKeyframe )
{
    if ( closestKeyframe == nullptr )
    {
        return Result::Failed;
    }

    m_alignment->align( closestKeyframe, m_curFrame );
    return Result::Success;
}

bool System::computeTrackingQuality( const std::shared_ptr< Frame >& frame, const uint32_t refFrameNumberObservations )
{
    const int32_t curNumberObservations = frame->numberObservation();
    if ( frame->numberObservation() < 50 )
    {
        return false;
    }
    const int32_t droppedFeatures = refFrameNumberObservations - curNumberObservations;
    if ( droppedFeatures > 40 )
    {
        return false;
    }
    return true;
}

bool System::needKeyframe( const Eigen::VectorXd& depthsInCurFrame, const double sceneDepthMean )
{
    // for ( const auto& frame : overlapKeyFrames )
    // {
    //     const Eigen::Vector3d diffPose = m_curFrame->world2camera( frame.first->cameraInWorld() );
    //     if ( std::abs( diffPose.x() ) / sceneDepthMean < 0.12 && std::abs( diffPose.y() ) / sceneDepthMean < 0.12 * 0.8 &&
    //          std::abs( diffPose.z() ) / sceneDepthMean < 0.12 * 1.3 )
    //         return false;
    // }
    // return true;

    const auto& relativePose     = algorithm::computeRelativePose( m_refFrame, m_curFrame );
    const double diffTranslation = relativePose.translation().norm();
    System_Log( DEBUG ) << "Diff Translation: " << diffTranslation;

    const uint64_t diffTime = m_curFrame->m_timestamp - m_curFrame->m_lastKeyframe->m_timestamp;
    System_Log( DEBUG ) << "Diff Timestamp: " << diffTime;

    const uint32_t diffId = m_curFrame->m_id - m_curFrame->m_lastKeyframe->m_id;
    System_Log( DEBUG ) << "Diff ID: " << diffId;

    const uint32_t numberVisiblePoints = m_curFrame->numberObservationWithPoints();
    System_Log( DEBUG ) << "Number Visible Pints: " << numberVisiblePoints;

    const uint32_t visibleFeatureInCurrentFrame = algorithm::computeNumberProjectedPoints( m_curFrame );
    const double featuresPropertion =
      static_cast< double >( visibleFeatureInCurrentFrame ) / m_curFrame->m_lastKeyframe->numberObservationWithPoints();
    System_Log( DEBUG ) << "feature Propertion: " << featuresPropertion;

    // if ( diffTranslation < 3.5 && diffId < 10 && numberVisiblePoints > 50 && featuresPropertion > 0.6 )
    // if ( diffTranslation < 3.5 && diffId < 10 && numberVisiblePoints > 50 )
    if ( diffId < 3 )
    {
        return true;
    }

    return false;
}

void System::reportSummary( const bool withDetail )
{
    std::set< std::shared_ptr< Point > > points;
    std::set< std::shared_ptr< Feature > > features;

    //-------------------------------------------------------------------------------
    //    Frame ID        Num Features        Num Points        Active Shared Pointer
    //-------------------------------------------------------------------------------

    for ( const auto& frame : m_allFrames )
    {
        for ( const auto& feature : frame->m_features )
        {
            if ( feature->m_point != nullptr )
            {
                points.insert( feature->m_point );
            }
            features.insert( feature );
        }
    }

    System_Log( INFO ) << " -------------------------------------------- Summary Frames -------------------------------------------- ";
    System_Log( INFO ) << "|                                                                                                         ";
    for ( const auto& frame : m_allFrames )
    {
        uint32_t cntFeatureWithPoints = frame->numberObservationWithPoints();
        int64_t LastKFId              = -1;
        if ( frame->m_lastKeyframe != nullptr )
        {
            LastKFId = frame->m_lastKeyframe->m_id;
        }
        System_Log( INFO ) << "| Frame ID: " << std::left << std::setw( 8 ) << frame->m_id << "KeyFrame: " << std::boolalpha
                           << frame->isKeyframe() << "\tNum Features: " << std::left << std::setw( 8 ) << frame->numberObservation()
                           << "Num Features With Points: " << std::left << std::setw( 8 ) << cntFeatureWithPoints
                           << "Last Key Frame id: " << std::left << std::setw( 8 ) << LastKFId << "Use Count: " << frame.use_count();
    }
    System_Log( INFO ) << "| Num Features: " << features.size();
    System_Log( INFO ) << "| Num Points: " << points.size();
    System_Log( INFO ) << "| Num Filters: " << m_depthEstimator->numberFilters();
    System_Log( INFO ) << " -------------------------------------------------------------------------------------------------------- \n";

    if ( withDetail == true )
    {
        System_Log( INFO ) << " ------------------------------------------- Summary Features ------------------------------------------- ";
        System_Log( INFO ) << "|                                                                                                         ";
        for ( const auto& frame : m_allFrames )
        {
            // System_Log( INFO ) << "|                                                                               |";
            System_Log( INFO ) << " ---------------------------------------------- Frame ID: " << frame->m_id
                               << " --------------------------------------------- ";
            // System_Log( INFO ) << "|                                                                               |";
            for ( const auto& feature : frame->m_features )
            {
                if ( feature->m_point != nullptr )
                {
                    System_Log( INFO ) << "| Feature ID: " << std::left << std::setw( 12 ) << feature->m_id << "Point ID: " << std::left
                                       << std::setw( 12 ) << feature->m_point->m_id << "Position: " << feature->m_pixelPosition.transpose()
                                       << "\t\tUse Count: " << std::left << std::setw( 6 ) << feature.use_count();
                    points.insert( feature->m_point );
                }
                else
                {
                    System_Log( INFO ) << "| Feature ID: " << std::left << std::setw( 12 ) << feature->m_id << "No Point\t\t  "
                                       << "Position: " << feature->m_pixelPosition.transpose() << "\t\tUse Count: " << std::left
                                       << std::setw( 6 ) << feature.use_count();
                }
            }
            System_Log( INFO )
              << " -------------------------------------------------------------------------------------------------------- \n";
            // System_Log( INFO );
        }

        System_Log( INFO ) << " -------------------------------- Summary Points ( " << points.size()
                           << " ) --------------------------------- ";
        System_Log( INFO ) << "|                                                                                                       ";
        for ( const auto& point : points )
        {
            System_Log( INFO ) << " --------------------------------------------- Point ID: " << point->m_id
                               << " --------------------------------------------- ";
            // System_Log( INFO ) << "|                                                                               |";

            System_Log( INFO ) << "| Num Features: " << std::left << std::setw( 10 ) << point->numberObservation()
                               << "Position: " << point->m_position.transpose() << "\t\tUse Count: " << std::left << std::setw( 6 )
                               << point.use_count();
            System_Log( INFO )
              << " -------------------------------------------------------------------------------------------------------- ";

            for ( const auto& feature : point->m_features )
            {
                System_Log( INFO ) << "| Frame ID: " << std::left << std::setw( 12 ) << feature->m_frame->m_id
                                   << "Feature ID: " << std::left << std::setw( 12 ) << feature->m_id
                                   << "Position: " << feature->m_pixelPosition.transpose();
            }
            System_Log( INFO )
              << " --------------------------------------------------------------------------------------------------------\n ";
        }
    }
}

bool System::loadCameraIntrinsics( const std::string& filename, cv::Mat& cameraMatrix, cv::Mat& distortionCoeffs )
{
    try
    {
        cv::FileStorage fs( filename, cv::FileStorage::READ );
        if ( !fs.isOpened() )
        {
            std::cout << "Failed to open " << filename << std::endl;
            return false;
        }

        fs[ "K" ] >> cameraMatrix;
        fs[ "d" ] >> distortionCoeffs;
        fs.release();
        return true;
    }
    catch ( std::exception& e )
    {
        std::cout << e.what() << std::endl;
        return false;
    }
}

void System::writeInFile( std::ofstream& fileWriter )
{
    const auto& extrinsic = m_refFrame->m_absPose.inverse().matrix3x4();
    // fileWriter << extrinsic << std::endl;
    fileWriter << std::setprecision( 6 ) << extrinsic.format( utils::eigenFormatIO() ) << std::endl;
    // fileWriter << extrinsic( 0, 0 ) << " " << extrinsic( 0, 1 ) << " " << extrinsic( 0, 2 ) << " " << extrinsic( 0, 3 ) << " "
    //            << extrinsic( 1, 0 ) << " " << extrinsic( 1, 1 ) << " " << extrinsic( 1, 2 ) << " " << extrinsic( 1, 3 ) << " "
    //            << extrinsic( 2, 0 ) << " " << extrinsic( 2, 1 ) << " " << extrinsic( 2, 2 ) << " " << extrinsic( 2, 3 ) << " "
    //            << extrinsic( 3, 0 ) << " " << extrinsic( 3, 1 ) << " " << extrinsic( 3, 2 ) << " " << extrinsic( 3, 3 ) << std::endl;
}