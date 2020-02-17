#include "system.hpp"
#include "algorithm.hpp"
#include "matcher.hpp"
#include "utils.hpp"
#include "visualization.hpp"

#include <chrono>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>

System::System( Config& config ) : m_config( &config )
{
    // const nlohmann::json& cameraJson  = jsonConfig[ "camera" ];
    // const std::string calibrationFile = utils::findAbsoluteFilePath( cameraJson[ "camera_calibration" ].get< std::string >() );
    // cv::Mat cameraMatrix;
    // cv::Mat distortionCoeffs;
    // bool result = loadCameraIntrinsics( calibrationFile, cameraMatrix, distortionCoeffs );
    // if ( result == false )
    // {
    //     std::cout << "Failed to open the calibration file, check config.json file" << std::endl;
    //     // return EXIT_FAILURE;
    // }

    // const int32_t imgWidth  = cameraJson[ "img_width" ].get< int32_t >();
    // const int32_t imgHeight = cameraJson[ "img_height" ].get< int32_t >();
    // m_config = Config::getInstance();
    // std::cout << "calibration: " << m_config->m_cameraCalibrationPath;
    const std::string calibrationFile = utils::findAbsoluteFilePath( m_config->m_cameraCalibrationPath );
    cv::Mat cameraMatrix;
    cv::Mat distortionCoeffs;
    loadCameraIntrinsics( calibrationFile, cameraMatrix, distortionCoeffs );
    m_camera  = std::make_shared< PinholeCamera >( m_config->m_imgWidth, m_config->m_imgHeight, cameraMatrix, distortionCoeffs );
    m_alignment = std::make_shared< ImageAlignment >( m_config->m_patchSizeImageAlignment, m_config->m_minLevelImagePyramid,
                                                    m_config->m_maxLevelImagePyramid, 6 );
}

void System::processFirstFrame( const cv::Mat& firstImg )
{
    m_refFrame = std::make_shared< Frame >( m_camera, firstImg );
    // Frame refFrame( camera, refImg );
    m_featureSelection = std::make_unique< FeatureSelection >( m_refFrame->m_imagePyramid.getBaseImage() );
    // FeatureSelection featureSelection( m_refFrame->m_imagePyramid.getBaseImage() );
    m_featureSelection->detectFeaturesInGrid( m_refFrame, m_config->m_gridPixelSize );

    m_refFrame->setKeyframe();
    std::cout << "Number of Features: " << m_refFrame->numberObservation() << std::endl;
    m_allKeyFrames.emplace_back( m_refFrame );
}

void System::processSecondFrame( const cv::Mat& secondImg )
{
    m_curFrame = std::make_shared< Frame >( m_camera, secondImg );

    Eigen::Matrix3d E;
    Eigen::Matrix3d F;
    Eigen::Matrix3d R;
    Eigen::Vector3d t;

    Matcher::computeOpticalFlowSparse( m_refFrame, m_curFrame, m_config->m_patchSizeOpticalFlow );
    Matcher::computeEssentialMatrix( m_refFrame, m_curFrame, 1.0, E );
    F = m_curFrame->m_camera->invK().transpose() * E * m_refFrame->m_camera->invK();
    algorithm::recoverPose( E, m_refFrame, m_curFrame, R, t );

    std::size_t numObserves = m_curFrame->numberObservation();
    Eigen::MatrixXd pointsWorld( 3, numObserves );
    Eigen::MatrixXd pointsCurCamera( 3, numObserves );
    Eigen::VectorXd depthCurFrame( numObserves );
    algorithm::points3DWorld( m_refFrame, m_curFrame, pointsWorld );
    algorithm::transferPointsWorldToCam( m_curFrame, pointsWorld, pointsCurCamera );
    algorithm::depthCamera( m_curFrame, pointsWorld, depthCurFrame );
    double medianDepth = algorithm::computeMedian( depthCurFrame );
    const double scale = 1.0 / medianDepth;
    m_curFrame->m_TransW2F.translation() =
      -m_curFrame->m_TransW2F.rotationMatrix() *
      ( m_refFrame->cameraInWorld() + scale * ( m_curFrame->cameraInWorld() - m_refFrame->cameraInWorld() ) );

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

    std::cout << "Points: " << pointsWorld.cols() << " cnt: " << cnt << " num ref observes: " << m_refFrame->numberObservation()
              << " num cur observes: " << m_curFrame->numberObservation() << std::endl;

    numObserves = m_refFrame->numberObservation();
    Eigen::VectorXd newCurDepths( numObserves );
    algorithm::depthCamera( m_curFrame, newCurDepths );
    medianDepth           = algorithm::computeMedian( newCurDepths );
    const double minDepth = newCurDepths.minCoeff();
    // std::cout << "Mean: " << medianDepth << " min: " << minDepth << std::endl;
    m_curFrame->setKeyframe();
    std::cout << "Number of Features: " << m_curFrame->numberObservation() << std::endl;
    m_allKeyFrames.emplace_back( m_curFrame );
}

void System::processNewFrame( const cv::Mat& newImg )
{
    // https://docs.microsoft.com/en-us/cpp/cpp/how-to-create-and-use-shared-ptr-instances?view=vs-2019
    m_refFrame = std::move( m_curFrame );
    // std::cout << "counter ref: " << m_refFrame.use_count() << std::endl;
    m_curFrame = std::make_shared< Frame >( m_camera, newImg );
    // std::cout << "counter cur: " << m_curFrame.use_count() << std::endl;

    // ImageAlignment match( m_config->m_patchSizeImageAlignment, m_config->m_minLevelImagePyramid, m_config->m_maxLevelImagePyramid, 6 );
    auto t1 = std::chrono::high_resolution_clock::now();
    m_alignment->align( m_refFrame, m_curFrame );
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "Elapsed time for alignment: " << std::chrono::duration_cast< std::chrono::microseconds >( t2 - t1 ).count()
              << " micro sec" << std::endl;
    {
        cv::Mat refBGR = visualization::getBGRImage( m_refFrame->m_imagePyramid.getBaseImage() );
        cv::Mat curBGR = visualization::getBGRImage( m_curFrame->m_imagePyramid.getBaseImage() );
        visualization::featurePoints( refBGR, m_refFrame, 11, "pink", visualization::drawingRectangle );
        // visualization::featurePointsInGrid(curBGR, curFrame, 50);
        // visualization::featurePoints(newBGR, newFrame);
        // visualization::project3DPoints(curBGR, curFrame);
        visualization::projectPointsWithRelativePose( curBGR, m_refFrame, m_curFrame, 8, "orange", visualization::drawingCircle );
        cv::Mat stickImg;
        visualization::stickTwoImageHorizontally( refBGR, curBGR, stickImg );
        cv::imshow( "both_image_1_2_optimization", stickImg );
        // cv::imshow("relative_1_2", curBGR);
    }

    for ( const auto& refFeatures : m_refFrame->m_frameFeatures )
    {
        const auto& point      = refFeatures->m_point->m_position;
        const auto& curFeature = m_curFrame->world2image( point );
        if ( m_curFrame->m_camera->isInFrame( curFeature, 5.0 ) == true )
        {
            std::unique_ptr< Feature > newFeature = std::make_unique< Feature >( m_curFrame, curFeature, 0.0 );
            m_curFrame->addFeature( newFeature );
            m_curFrame->m_frameFeatures.back()->setPoint( refFeatures->m_point );
        }
    }
    std::cout << "Number of Features: " << m_curFrame->numberObservation() << std::endl;
    m_allKeyFrames.emplace_back( m_curFrame );
}

void System::reportSummaryFrames()
{
    //-------------------------------------------------------------------------------
    //    Frame ID        Num Features        Num Points        Active Shared Pointer
    //-------------------------------------------------------------------------------

    std::cout << " ----------------------------- Report Summary Frames -------------------------------- " << std::endl;
    std::cout << "|                                                                                    |" << std::endl;
    for ( const auto& frame : m_allKeyFrames )
    {
        std::cout << "| Frame ID: " << frame->m_id << "\t\t"
                  << "Num Feaures: " << frame->numberObservation() << "\t\t"
                  << "Active Shared Pointer: " << frame.use_count() << "     |" << std::endl;
    }
    std::cout << "|                                                                                    |" << std::endl;
    std::cout << " ------------------------------------------------------------------------------------ " << std::endl;
}

void System::reportSummaryFeatures()
{
    std::cout << " -------------------------- Report Summary Features ---------------------------- " << std::endl;
    for ( const auto& frame : m_allKeyFrames )
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