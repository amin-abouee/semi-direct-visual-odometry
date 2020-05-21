#include "system.hpp"
#include "algorithm.hpp"
#include "utils.hpp"
#include "visualization.hpp"

#include <chrono>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>

#include "easylogging++.h"
#define System_Log( LEVEL ) CLOG( LEVEL, "System" )

// System::System( const Config& config )
System::System( const Config& config ) : m_config( &config )
{
    const std::string calibrationFile = utils::findAbsoluteFilePath( m_config->m_cameraCalibrationPath );
    cv::Mat cameraMatrix;
    cv::Mat distortionCoeffs;
    loadCameraIntrinsics( calibrationFile, cameraMatrix, distortionCoeffs );
    m_camera    = std::make_shared< PinholeCamera >( m_config->m_imgWidth, m_config->m_imgHeight, cameraMatrix, distortionCoeffs );
    m_alignment = std::make_shared< ImageAlignment >( m_config->m_patchSizeImageAlignment, m_config->m_minLevelImagePyramid,
                                                      m_config->m_maxLevelImagePyramid, 6 );
}

void System::processFirstFrame( const cv::Mat& firstImg )
{
    m_refFrame = std::make_shared< Frame >( m_camera, firstImg, m_config->m_maxLevelImagePyramid + 1 );
    // Frame refFrame( camera, refImg );
    m_featureSelection = std::make_unique< FeatureSelection >( m_refFrame->m_imagePyramid.getBaseImage() );
    // FeatureSelection featureSelection( m_refFrame->m_imagePyramid.getBaseImage() );
    m_featureSelection->detectFeaturesInGrid( m_refFrame, m_config->m_gridPixelSize );
    // m_featureSelection->detectFeaturesByValue( m_refFrame, 150 );
    // m_featureSelection->detectFeaturesWithSSC(m_refFrame, 1000);

    // {
    //     cv::Mat gradient = m_featureSelection->m_gradientMagnitude.clone();
    //     cv::normalize(gradient, gradient, 0, 255, cv::NORM_MINMAX, CV_8U);
    //     cv::Mat refBGR = visualization::getBGRImage( gradient );
    //     // cv::Mat refBGR = visualization::getBGRImage( m_refFrame->m_imagePyramid.getBaseImage() );
    //     visualization::featurePoints( refBGR, m_refFrame, 5, "pink", visualization::drawingRectangle );
    //     // visualization::imageGrid(refBGR, m_refFrame, m_config->m_gridPixelSize, "amber");
    //     cv::imshow("ref", refBGR);
    //     cv::waitKey(0);
    // }

    m_refFrame->setKeyframe();
    System_Log( DEBUG ) << "Number of Features: " << m_refFrame->numberObservation();
    m_keyFrames.emplace_back( m_refFrame );
}

void System::processSecondFrame( const cv::Mat& secondImg )
{
    m_curFrame = std::make_shared< Frame >( m_camera, secondImg, m_config->m_maxLevelImagePyramid + 1 );

    Eigen::Matrix3d E;
    Eigen::Matrix3d F;
    Eigen::Matrix3d R;
    Eigen::Vector3d t;

    algorithm::computeOpticalFlowSparse( m_refFrame, m_curFrame, m_config->m_patchSizeOpticalFlow );
    algorithm::computeEssentialMatrix( m_refFrame, m_curFrame, 1.0, E );
    // F = m_curFrame->m_camera->invK().transpose() * E * m_refFrame->m_camera->invK();

    // decompose essential matrix to R and t and set the absolute pose of reference frame
    algorithm::recoverPose( E, m_refFrame, m_curFrame, R, t );

    std::size_t numObserves = m_curFrame->numberObservation();
    Eigen::MatrixXd pointsWorld( 3, numObserves );
    Eigen::MatrixXd pointsCurCamera( 3, numObserves );
    Eigen::VectorXd depthCurFrame( numObserves );

    algorithm::triangulate3DWorldPoints( m_refFrame, m_curFrame, pointsWorld );
    algorithm::transferPointsWorldToCam( m_curFrame, pointsWorld, pointsCurCamera );
    algorithm::depthCamera( m_curFrame, pointsWorld, depthCurFrame );
    double medianDepth = algorithm::computeMedian( depthCurFrame );
    System_Log( DEBUG ) << "Median depth in current frame: " << medianDepth;
    const double scale = 1.0 / medianDepth;

    // t = -RC
    // In this formula, we assume, the W = K_{1}, means the m_refFrame->cameraInWorld() = (0, 0, 0)
    // first we compute the C, scale it and compute t
    m_curFrame->m_TransW2F.translation() =
      -m_curFrame->m_TransW2F.rotationMatrix() *
      ( m_refFrame->cameraInWorld() + scale * ( m_curFrame->cameraInWorld() - m_refFrame->cameraInWorld() ) );

    // scale current frame depth
    pointsCurCamera *= scale;
    algorithm::transferPointsCamToWorld( m_curFrame, pointsCurCamera, pointsWorld );

    uint32_t cnt = 0;
    for ( std::size_t i( 0 ); i < numObserves; i++ )
    {
        const Eigen::Vector2d refFeature = m_refFrame->m_frameFeatures[ i ]->m_feature;
        const Eigen::Vector2d curFeature = m_curFrame->m_frameFeatures[ i ]->m_feature;
        if ( m_refFrame->m_camera->isInFrame( refFeature, 5.0 ) == true && m_curFrame->m_camera->isInFrame( curFeature, 5.0 ) == true &&
             pointsCurCamera.col( i ).z() > 0 )
        {
            std::shared_ptr< Point > point = std::make_shared< Point >( pointsWorld.col( i ) );
            //    std::cout << "3D points: " << point->m_position.format(utils::eigenFormat()) << std::endl;
            m_refFrame->m_frameFeatures[ i ]->setPoint( point );
            m_curFrame->m_frameFeatures[ i ]->setPoint( point );
            cnt++;
        }
    }

    m_refFrame->m_frameFeatures.erase( std::remove_if( m_refFrame->m_frameFeatures.begin(), m_refFrame->m_frameFeatures.end(),
                                                       []( const auto& feature ) { return feature->m_point == nullptr; } ),
                                       m_refFrame->m_frameFeatures.end() );

    m_curFrame->m_frameFeatures.erase( std::remove_if( m_curFrame->m_frameFeatures.begin(), m_curFrame->m_frameFeatures.end(),
                                                       []( const auto& feature ) { return feature->m_point == nullptr; } ),
                                       m_curFrame->m_frameFeatures.end() );

    System_Log( INFO ) << "Points: " << pointsWorld.cols() << " cnt: " << cnt << " num ref observes: " << m_refFrame->numberObservation()
                       << " num cur observes: " << m_curFrame->numberObservation();

    numObserves = m_refFrame->numberObservation();
    Eigen::VectorXd newCurDepths( numObserves );
    algorithm::depthCamera( m_curFrame, newCurDepths );
    medianDepth = algorithm::computeMedian( newCurDepths );
    // const double minDepth = newCurDepths.minCoeff();
    // std::cout << "Mean: " << medianDepth << " min: " << minDepth << std::endl;
    m_curFrame->setKeyframe();
    System_Log( INFO ) << "Number of Features: " << m_curFrame->numberObservation();
    m_keyFrames.emplace_back( m_curFrame );
}

void System::processNewFrame( const cv::Mat& newImg )
{
    // https://docs.microsoft.com/en-us/cpp/cpp/how-to-create-and-use-shared-ptr-instances?view=vs-2019
    m_refFrame = std::move( m_curFrame );
    // std::cout << "counter ref: " << m_refFrame.use_count() << std::endl;
    m_curFrame = std::make_shared< Frame >( m_camera, newImg, m_config->m_maxLevelImagePyramid + 1 );
    // std::cout << "counter cur: " << m_curFrame.use_count() << std::endl;

    // ImageAlignment match( m_config->m_patchSizeImageAlignment, m_config->m_minLevelImagePyramid, m_config->m_maxLevelImagePyramid, 6 );
    auto t1 = std::chrono::high_resolution_clock::now();
    m_alignment->align( m_refFrame, m_curFrame );
    auto t2 = std::chrono::high_resolution_clock::now();
    System_Log( INFO ) << "Elapsed time for alignment: " << std::chrono::duration_cast< std::chrono::microseconds >( t2 - t1 ).count()
                       << " micro sec";
    {
        cv::Mat refBGR = visualization::getBGRImage( m_refFrame->m_imagePyramid.getBaseImage() );
        cv::Mat curBGR = visualization::getBGRImage( m_curFrame->m_imagePyramid.getBaseImage() );
        visualization::featurePoints( refBGR, m_refFrame, 8, "pink", visualization::drawingRectangle );
        // visualization::featurePointsInGrid(curBGR, curFrame, 50);
        // visualization::featurePoints(newBGR, newFrame);
        // visualization::project3DPoints(curBGR, curFrame);
        visualization::projectPointsWithRelativePose( curBGR, m_refFrame, m_curFrame, 8, "orange", visualization::drawingCircle );
        cv::Mat stickImg;
        visualization::stickTwoImageHorizontally( refBGR, curBGR, stickImg );
        cv::imshow( "both_image_1_2_optimization", stickImg );
        // cv::imshow("relative_1_2", curBGR);
        cv::waitKey( 0 );
    }

    for ( const auto& refFeatures : m_refFrame->m_frameFeatures )
    {
        const auto& point      = refFeatures->m_point->m_position;
        const auto& curFeature = m_curFrame->world2image( point );
        if ( m_curFrame->m_camera->isInFrame( curFeature, 5.0 ) == true )
        {
            std::shared_ptr< Feature > newFeature = std::make_shared< Feature >( m_curFrame, curFeature, 0.0 );
            m_curFrame->addFeature( newFeature );
            m_curFrame->m_frameFeatures.back()->setPoint( refFeatures->m_point );
        }
    }
    System_Log( INFO ) << "Number of Features: " << m_curFrame->numberObservation();
    m_keyFrames.emplace_back( m_curFrame );
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
        for ( const auto& feature : frame->m_frameFeatures )
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