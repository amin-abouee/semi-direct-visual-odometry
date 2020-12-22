#include "system.hpp"
#include "algorithm.hpp"
#include "utils.hpp"
#include "visualization.hpp"

#include <chrono>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>

#include "easylogging++.h"
#define System_Log( LEVEL ) CLOG( LEVEL, "System" )

System::System( const std::shared_ptr< Config >& config ) : m_config( config ), m_systemStatus( System::Status::Process_First_Frame )
{
    const std::string calibrationFile = utils::findAbsoluteFilePath( m_config->m_cameraCalibrationPath );
    cv::Mat cameraMatrix;
    cv::Mat distortionCoeffs;
    loadCameraIntrinsics( calibrationFile, cameraMatrix, distortionCoeffs );
    m_camera          = std::make_shared< PinholeCamera >( m_config->m_imgWidth, m_config->m_imgHeight, cameraMatrix, distortionCoeffs );
    m_alignment       = std::make_shared< ImageAlignment >( m_config->m_patchSizeImageAlignment, m_config->m_minLevelImagePyramid,
                                                      m_config->m_maxLevelImagePyramid, 6 );
    m_featureSelector = std::make_shared< FeatureSelection >( m_config->m_imgWidth, m_config->m_imgHeight, m_config->m_cellPixelSize );
    m_depthEstimator  = std::make_unique< DepthEstimator >( m_featureSelector );
    m_map             = std::make_unique< Map >( m_camera, 32 );
    m_bundler         = std::make_shared< BundleAdjustment >( 0, 6 );
}

void System::addImage( const cv::Mat& img, const uint64_t timestamp )
{
    m_curFrame = std::make_shared< Frame >( m_camera, img, m_config->m_maxLevelImagePyramid + 1, timestamp );
    System_Log( INFO ) << "Processing frame id: " << m_curFrame->m_id;
    System::Result res;

    if ( m_systemStatus == System::Status::Procese_New_Frame )
    {
        res = processNewFrame();
    }
    else if ( m_systemStatus == System::Status::Process_Second_Frame )
    {
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

    m_refFrame = std::move( m_curFrame );
}

System::Result System::processFirstFrame()
{
    TIMED_FUNC( timerFirstImage );
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
        cv::Mat refBGR   = visualization::getBGRImage( gradient );
        visualization::featurePoints( refBGR, m_curFrame, 5, "pink", visualization::drawingRectangle );
        visualization::imageGrid( refBGR, m_config->m_cellPixelSize, "amber" );
        cv::imshow( "First Image", refBGR );
    }

    m_curFrame->setKeyframe();
    m_map->addKeyframe( m_curFrame );
    System_Log( DEBUG ) << "Number of Features: " << m_curFrame->numberObservation();
    m_systemStatus = System::Status::Process_Second_Frame;
    return Result::Success;
}

System::Result System::processSecondFrame()
{
    TIMED_FUNC( timerSecondImage );
    System_Log( DEBUG ) << "Number of Features: " << m_refFrame->numberObservation();

    Eigen::Matrix3d E;
    Eigen::Matrix3d F;
    Eigen::Matrix3d R;
    Eigen::Vector3d t;

    // we find teh corresponding point in curFrame and create feature for them
    algorithm::computeOpticalFlowSparse( m_refFrame, m_curFrame, m_config->m_patchSizeOpticalFlow );
    algorithm::computeEssentialMatrix( m_refFrame, m_curFrame, 1.0, E );

    // TODO: check the number of matched!!!

    // decompose essential matrix to R and t and set the absolute pose of reference frame
    algorithm::recoverPose( E, m_refFrame, m_curFrame, R, t );
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
    algorithm::depthCamera( m_curFrame, pointsWorld, depthCurFrame );
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
            //    std::cout << "3D points: " << point->m_position.format(utils::eigenFormat()) << std::endl;
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

    // FIXME: do BA

    m_curFrame->setKeyframe();
    numObserves = m_curFrame->numberObservation();
    Eigen::VectorXd newCurDepths( numObserves );
    algorithm::depthCamera( m_curFrame, newCurDepths );
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
    // m_keyFrames.emplace_back( m_curFrame );
    // m_depthEstimator->addKeyframe(m_curFrame, medianDepth, 0.5 * minDepth);
    m_map->addKeyframe( m_curFrame );

    if ( m_config->m_enableVisualization == true )
    {
        cv::Mat refBGR = visualization::getBGRImage( m_refFrame->m_imagePyramid.getBaseImage() );
        cv::Mat curBGR = visualization::getBGRImage( m_curFrame->m_imagePyramid.getBaseImage() );
        visualization::featurePoints( refBGR, m_refFrame, 8, "pink", visualization::drawingRectangle );
        visualization::projectPointsWithRelativePose( curBGR, m_refFrame, m_curFrame, 8, "orange", visualization::drawingCircle );
        cv::Mat stickImg;
        visualization::stickTwoImageHorizontally( refBGR, curBGR, stickImg );
        cv::Mat curBGR2 = visualization::getBGRImage( m_curFrame->m_imagePyramid.getBaseImage() );
        visualization::featurePoints( curBGR2, m_curFrame, 8, "cyan", visualization::drawingCircle );
        cv::Mat stickBigImg;
        visualization::stickTwoImageHorizontally( stickImg, curBGR2, stickBigImg );

        std::stringstream ss;
        ss << m_refFrame->m_id << " -> " << m_curFrame->m_id;
        cv::imshow( ss.str(), stickImg );
        cv::imshow( "relative_1_2", curBGR );

        cv::Mat imageDepth = m_featureSelector->m_imgGradientMagnitude.clone();
        visualization::colormapDepth( imageDepth, m_curFrame, 7, "amber" );
        cv::imshow( "depth", imageDepth );
    }

    m_systemStatus = System::Status::Procese_New_Frame;
    return Result::Success;
}

System::Result System::processNewFrame()
{
    TIMED_FUNC( timerNewFrame );
    // https://docs.microsoft.com/en-us/cpp/cpp/how-to-create-and-use-shared-ptr-instances?view=vs-2019
    // m_refFrame = std::move( m_curFrame );
    // std::cout << "counter ref: " << m_refFrame.use_count() << std::endl;
    // m_curFrame = std::make_shared< Frame >( m_camera, newImg, m_config->m_maxLevelImagePyramid + 1 );
    // std::cout << "counter cur: " << m_curFrame.use_count() << std::endl;
    m_curFrame->m_absPose = m_refFrame->m_absPose;
    System_Log( INFO ) << "Number of features, pre: " << m_refFrame->numberObservation() << ", cur: " << m_curFrame->numberObservation();

    // ImageAlignment match( m_config->m_patchSizeImageAlignment, m_config->m_minLevelImagePyramid, m_config->m_maxLevelImagePyramid, 6 );
    auto t1 = std::chrono::high_resolution_clock::now();
    m_alignment->align( m_refFrame, m_curFrame );
    auto t2 = std::chrono::high_resolution_clock::now();
    System_Log( INFO ) << "Elapsed time for alignment: " << std::chrono::duration_cast< std::chrono::microseconds >( t2 - t1 ).count()
                       << " micro sec";
    {
        // cv::Mat refBGR = visualization::getBGRImage( m_refFrame->m_imagePyramid.getBaseImage() );
        // cv::Mat curBGR = visualization::getBGRImage( m_curFrame->m_imagePyramid.getBaseImage() );
        // visualization::featurePoints( refBGR, m_refFrame, 8, "pink", visualization::drawingRectangle );
        // // visualization::featurePointsInGrid(curBGR, curFrame, 50);
        // // visualization::featurePoints(newBGR, newFrame);
        // // visualization::project3DPoints(curBGR, curFrame);
        // visualization::projectPointsWithRelativePose( curBGR, m_refFrame, m_curFrame, 8, "orange", visualization::drawingRectangle );
        // cv::Mat stickImg;
        // visualization::stickTwoImageHorizontally( refBGR, curBGR, stickImg );
        // std::stringstream ss;
        // ss << m_refFrame->m_id << " -> " << m_curFrame->m_id;
        // cv::imshow( ss.str(), stickImg );
        // cv::imshow( "tracking", stickImg );
        // cv::imshow("relative_1_2", curBGR);
        // cv::waitKey( 0 );
        // System_Log( INFO ) << "ref id: " << m_refFrame->m_id << ", cur id: " << m_curFrame->m_id;
    }

    // for ( const auto& refFeatures : m_refFrame->m_features )
    // {
    //     if ( refFeatures->m_point == nullptr )
    //     {
    //         continue;
    //     }

    //     const auto& point      = refFeatures->m_point->m_position;
    //     const auto& curFeature = m_curFrame->world2image( point );
    //     if ( m_curFrame->m_camera->isInFrame( curFeature, 5.0 ) == true )
    //     {
    //         std::shared_ptr< Feature > newFeature = std::make_shared< Feature >( m_curFrame, curFeature, 0.0 );
    //         m_curFrame->addFeature( newFeature );
    //         m_curFrame->m_features.back()->setPoint( refFeatures->m_point );
    //     }
    // }
    std::vector< frameSize > overlapKeyFrames;
    m_map->reprojectMap( m_curFrame, overlapKeyFrames );
    System_Log( INFO ) << "Number of Features: " << m_curFrame->numberObservation();

    if ( m_map->m_matches < 50 )
    {
        m_curFrame->m_absPose = m_refFrame->m_absPose;
        return Result::Failed;
    }

    // select keyframe
    // core_kfs_.insert(new_frame_);
    // setTrackingQuality(sfba_n_edges_final);
    // if(tracking_quality_ == TRACKING_INSUFFICIENT)
    // {
    //     new_frame_->T_f_w_ = last_frame_->T_f_w_; // reset to avoid crazy pose jumps
    //     return RESULT_FAILURE;
    // }

    m_bundler->optimizePose( m_curFrame );
    m_bundler->optimizeStructure( m_curFrame, 50 );
    m_keyFrames.emplace_back( m_curFrame );

    const uint32_t numObserves = m_curFrame->numberObservation();
    Eigen::VectorXd newCurDepths( numObserves );
    algorithm::depthCamera( m_curFrame, newCurDepths );
    const double depthMean = algorithm::computeMedian( newCurDepths );
    const double depthMin  = newCurDepths.minCoeff();

    if ( needKeyframe( depthMean, overlapKeyFrames ) )
    {
        m_depthEstimator->addFrame( m_curFrame );
        return Result::Success;
    }
    m_curFrame->setKeyframe();

    // TODO: add candidatepoint to map

    // TODO: use bundle adjustment

    m_depthEstimator->addKeyframe( m_curFrame, depthMean, depthMin * 0.5 );

    if ( m_map->m_keyFrames.size() > 10 )
    {
        std::shared_ptr< Frame > furthestFrame{ nullptr };
        m_map->getFurthestKeyframe( m_curFrame->cameraInWorld(), furthestFrame );
        m_depthEstimator->removeKeyframe( furthestFrame );
        m_map->removeFrame( furthestFrame );
    }

    m_map->addKeyframe( m_curFrame );
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

void System::reportSummaryFrames()
{
    //-------------------------------------------------------------------------------
    //    Frame ID        Num Features        Num Points        Active Shared Pointer
    //-------------------------------------------------------------------------------

    std::cout << " ----------------------------- Report Summary Frames -------------------------------- " << std::endl;
    std::cout << "|                                                                                    |" << std::endl;
    for ( const auto& frame : m_keyFrames )
    {
        std::cout << "| Frame ID: " << frame->m_id << "\t\t"
                  << "Num Features: " << frame->numberObservation() << "\t\t"
                  << "Active Shared Pointer: " << frame.use_count() << "     |" << std::endl;
    }
    std::cout << "|                                                                                    |" << std::endl;
    std::cout << " ------------------------------------------------------------------------------------ " << std::endl;
}

void System::reportSummaryFeatures()
{
    std::cout << " -------------------------- Report Summary Features ---------------------------- " << std::endl;
    for ( const auto& frame : m_keyFrames )
    {
        std::cout << "|                                                                               |" << std::endl;
        std::cout << " -------------------------------- Frame ID: " << frame->m_id << " ---------------------------------- " << std::endl;
        std::cout << "|                                                                               |" << std::endl;
        for ( const auto& feature : frame->m_features )
        {
            std::cout << "| Feature ID: " << std::left << std::setw( 12 ) << feature->m_id << "Point ID: " << std::left << std::setw( 12 )
                      << feature->m_point->m_id << "Cnt Shared Pointer Point: " << std::left << std::setw( 6 )
                      << feature->m_point.use_count() << "|" << std::endl;
        }
        std::cout << " ------------------------------------------------------------------------------- " << std::endl;
    }
}

void System::reportSummaryPoints()
{
}

// void System::makeKeyframe( std::shared_ptr< Frame >& frame, const double& depthMean, const double& depthMin )
// {
//     return;
// }

bool System::needKeyframe( const double sceneDepthMean, const std::vector< frameSize >& overlapKeyFrames )
{
    for ( const auto& frame : overlapKeyFrames )
    {
        const Eigen::Vector3d diffPose = m_curFrame->world2camera( frame.first->cameraInWorld() );
        if ( std::abs( diffPose.x() ) / sceneDepthMean < 0.12 && std::abs( diffPose.y() ) / sceneDepthMean < 0.12 * 0.8 &&
             std::abs( diffPose.z() ) / sceneDepthMean < 0.12 * 1.3 )
            return false;
    }
    return true;
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