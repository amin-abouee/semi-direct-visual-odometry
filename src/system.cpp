#include "system.hpp"
#include "utils.hpp"
#include "matcher.hpp"
#include "algorithm.hpp"
#include "image_alignment.hpp"
#include "visualization.hpp"

#include <chrono>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>

System::System(Config& config): m_config(&config)
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
    std::cout << "calibration: " << m_config->m_cameraCalibrationPath;
    const std::string calibrationFile = utils::findAbsoluteFilePath( m_config->m_cameraCalibrationPath );
    cv::Mat cameraMatrix;
    cv::Mat distortionCoeffs;
    loadCameraIntrinsics( calibrationFile, cameraMatrix, distortionCoeffs );
    m_camera = std::make_shared<PinholeCamera>( m_config->m_imgWidth, m_config->m_imgHeight, cameraMatrix, distortionCoeffs );
}

void System::processFirstFrame(const cv::Mat& firstImg)
{
    m_refFrame = std::make_shared<Frame>(*m_camera, firstImg);
    // Frame refFrame( camera, refImg );
    m_featureSelection = std::make_unique<FeatureSelection>(m_refFrame->m_imagePyramid.getBaseImage());
    // FeatureSelection featureSelection( m_refFrame->m_imagePyramid.getBaseImage() );
    m_featureSelection->detectFeaturesInGrid( *m_refFrame, m_config->m_gridPixelSize );

    m_refFrame->setKeyframe();
}

void System::processSecondFrame(const cv::Mat& secondImg)
{
    m_curFrame = std::make_shared<Frame>(*m_camera, secondImg);

    Eigen::Matrix3d E;
    Eigen::Matrix3d F;
    Eigen::Matrix3d R;
    Eigen::Vector3d t;

    Matcher::computeOpticalFlowSparse( *m_refFrame, *m_curFrame, m_config->m_patchSizeOpticalFlow );
    Matcher::computeEssentialMatrix( *m_refFrame, *m_curFrame, 1.0, E );
    F = m_curFrame->m_camera->invK().transpose() * E * m_refFrame->m_camera->invK();
    algorithm::recoverPose( E, *m_refFrame, *m_curFrame, R, t );


    std::size_t numObserves = m_curFrame->numberObservation();
    Eigen::MatrixXd pointsWorld( 3, numObserves );
    Eigen::MatrixXd pointsCurCamera( 3, numObserves );
    Eigen::VectorXd depthCurFrame( numObserves );
    algorithm::points3DWorld( *m_refFrame, *m_curFrame, pointsWorld );
    algorithm::transferPointsWorldToCam( *m_curFrame, pointsWorld, pointsCurCamera );
    algorithm::depthCamera( *m_curFrame, pointsWorld, depthCurFrame );
    double medianDepth = algorithm::computeMedian( depthCurFrame );
    const double scale = 1.0 / medianDepth;
    m_curFrame->m_TransW2F.translation() = -m_curFrame->m_TransW2F.rotationMatrix() *
                                    ( m_refFrame->cameraInWorld() + scale * ( m_curFrame->cameraInWorld() - m_refFrame->cameraInWorld() ) );

    pointsCurCamera *= scale;
    algorithm::transferPointsCamToWorld( *m_curFrame, pointsCurCamera, pointsWorld );
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
    algorithm::depthCamera( *m_curFrame, newCurDepths );
    medianDepth           = algorithm::computeMedian( newCurDepths );
    const double minDepth = newCurDepths.minCoeff();
    // std::cout << "Mean: " << medianDepth << " min: " << minDepth << std::endl;
    m_curFrame->setKeyframe();
}

void System::processNextFrame(const cv::Mat& newImg)
{
    m_newFrame = std::make_shared<Frame>(*m_camera, newImg);
    ImageAlignment match( m_config->m_patchSizeImageAlignment, m_config->m_minLevelImagePyramid, m_config->m_maxLevelImagePyramid, 6 );
    auto t1 = std::chrono::high_resolution_clock::now();
    match.align( *m_curFrame, *m_newFrame );
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "Elapsed time for alignment: " << std::chrono::duration_cast< std::chrono::microseconds >( t2 - t1 ).count() << " micro sec"
              << std::endl;
    {
        cv::Mat curBGR = visualization::getBGRImage( m_curFrame->m_imagePyramid.getBaseImage() );
        cv::Mat newBGR = visualization::getBGRImage( m_newFrame->m_imagePyramid.getBaseImage() );
        visualization::featurePoints( curBGR, *m_curFrame );
        // visualization::featurePointsInGrid(curBGR, curFrame, 50);
        // visualization::featurePoints(newBGR, newFrame);
        // visualization::project3DPoints(curBGR, curFrame);
        visualization::projectPointsWithRelativePose( newBGR, *m_curFrame, *m_newFrame );
        cv::Mat stickImg;
        visualization::stickTwoImageHorizontally( curBGR, newBGR, stickImg );
        cv::imshow( "both_image_1_2_optimization", stickImg );
        cv::imshow("relative_1_2", newBGR);
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