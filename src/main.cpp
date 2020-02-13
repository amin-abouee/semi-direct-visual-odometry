// #define EIGEN_DEFAULT_DENSE_INDEX_TYPE long
// #define EIGEN_DEFAULT_IO_FORMAT Eigen::IOFormat( 10, Eigen::DontAlignCols, ", ", " , ", "[", "]", "[", "]" )

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
// #include <opencv2/core/eigen.hpp>

#include <Eigen/Core>

#include "algorithm.hpp"
// #include "feature_selection.hpp"
#include "system.hpp"
// #include "image_alignment.hpp"
// #include "matcher.hpp"
// #include "point.hpp"
#include "utils.hpp"
// #include "visualization.hpp"
// #include "config.hpp"

// #include "spdlog/sinks/stdout_color_sinks.h"
#include <spdlog/spdlog.h>

#include <nlohmann/json.hpp>



int main( int argc, char* argv[] )
{
    int * myPointer = new int(5);
    std::shared_ptr<int> mySharedPtrA(myPointer);
    std::cout << "mySharedPtrA: " << mySharedPtrA.use_count() << std::endl;
    std::shared_ptr<int> mySharedPtrB = mySharedPtrA;
    std::cout << "mySharedPtrB: " << mySharedPtrB.use_count() << std::endl;

// System system();

#ifdef EIGEN_MALLOC_ALREADY_ALIGNED
    std::cout << "EIGEN_MALLOC_ALREADY_ALIGNED" << std::endl;
#endif

#ifdef EIGEN_FAST_MATH
    std::cout << "EIGEN_FAST_MATH" << std::endl;
#endif

#ifdef EIGEN_USE_MKL_ALL
    std::cout << "EIGEN_USE_MKL_ALL" << std::endl;
#endif

#ifdef EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    std::cout << "EIGEN_MAKE_ALIGNED_OPERATOR_NEW" << std::endl;
#endif

#ifdef EIGEN_USE_BLAS
    std::cout << "EIGEN_USE_BLAS" << std::endl;
#endif
    std::cout << "EIGEN_HAS_CXX11_CONTAINERS: " << EIGEN_HAS_CXX11_CONTAINERS << std::endl;
    std::cout << "EIGEN_MAX_CPP_VER: " << EIGEN_MAX_CPP_VER << std::endl;
    std::cout << "EIGEN_HAS_CXX11_MATH: " << EIGEN_HAS_CXX11_MATH << std::endl;
    // std::cout << "Number of Threads: " << Eigen::nbThreads( ) << std::endl;
    // Eigen::setNbThreads(4);
    // std::cout << "Number of Threads: " << Eigen::nbThreads( ) << std::endl;

    // auto mainLogger = spdlog::stdout_color_mt( "main" );
    // mainLogger->set_level( spdlog::level::debug );
    // mainLogger->set_pattern( "[%Y-%m-%d %H:%M:%S] [%s:%#] [%n->%l] [thread:%t] %v" );
    // mainLogger->info( "Info" );
    // mainLogger->debug( "Debug" );
    // mainLogger->warn( "Warn" );
    // mainLogger->error( "Error" );

    std::string configIOFile;
    if ( argc > 1 )
        configIOFile = argv[ 1 ];
    else
        configIOFile = "config/config.json";

    // Config::init(utils::findAbsoluteFilePath( configIOFile ));
    // Config* config = Config::getInstance();
    Config config (utils::findAbsoluteFilePath( configIOFile ));
    System system(config);

    // std::string oma = utils::findAbsoluteFilePath(configIOFile);

    // const nlohmann::json& configJson = createConfigParser( utils::findAbsoluteFilePath( configIOFile ) );
    // std::cout << configJson[ "file_paths" ][ "camera_calibration" ].get< std::string >() << std::endl;
    // std::ifstream jsonFile(configFile);

    // const nlohmann::json& cameraJson  = configJson[ "camera" ];
    // const std::string calibrationFile = utils::findAbsoluteFilePath( cameraJson[ "camera_calibration" ].get< std::string >() );
    // cv::Mat cameraMatrix;
    // cv::Mat distortionCoeffs;
    // bool result = loadCameraIntrinsizcs( calibrationFile, cameraMatrix, distortionCoeffs );
    // if ( result == false )
    // {
    //     std::cout << "Failed to open the calibration file, check config.json file" << std::endl;
    //     return EXIT_FAILURE;
    // }

    // const int32_t imgWidth  = cameraJson[ "img_width" ].get< int32_t >();
    // const int32_t imgHeight = cameraJson[ "img_height" ].get< int32_t >();

    const cv::Mat refImg = cv::imread( utils::findAbsoluteFilePath( "input/0000000000.png" ), cv::IMREAD_GRAYSCALE );
    const cv::Mat curImg = cv::imread( utils::findAbsoluteFilePath( "input/0000000001.png" ), cv::IMREAD_GRAYSCALE );

    // Eigen::Matrix3d K;
    // K << 7.215377e+02, 0.000000e+00, 6.095593e+02, 0.000000e+00, 7.215377e+02, 1.728540e+02, 0.000000e+00, 0.000000e+00, 1.000000e+00;
    // // std::cout << "Camera Matrix: \n" << K << std::endl;

    // Eigen::Matrix3d E;
    // E << .22644456e-03, -7.06943058e-01, -4.05822481e-03, 7.06984545e-01, 1.22048201e-03, 1.26855863e-02, 3.25653616e-03,
    // -1.46073125e-02,
    //   -2.59077801e-05;
    // // std::cout << "Old E: " << E.format( utils::eigenFormat() ) << std::endl;

    // Eigen::Matrix3d F;
    // F << -5.33286713e-08, -1.49632194e-03, 2.67961447e-01, 1.49436356e-03, -2.27291565e-06, -9.03327631e-01, -2.68937438e-01,
    //   9.02739500e-01, 1.00000000e+00;
    // // std::cout << "Fundamental Matrix: \n" << F << std::endl;

    // Eigen::Vector3d C( -0.01793327, -0.00577164, 1 );

    // // Do the cheirality check.
    // Eigen::Matrix3d R;
    // R << 0.99999475, 0.0017505, -0.0027263, -0.00174731, 0.99999779, 0.00117013, 0.00272834, -0.00116536, 0.9999956;
    // std::cout << "Old R1: " << R.format( utils::eigenFormat() ) << std::endl;

    // Eigen::Matrix3d R;
    // R << -0.99925367, -0.00151199, -0.03859818, 0.00191117, -0.99994505, -0.01030721, -0.03858047, -0.01037328,
    // 0.99920165;

    // Eigen::Vector3d t( -0.0206659, -0.00456935, 0.999776 );
    // Eigen::Vector3d t( 0.0206659, 0.00456935, -0.999776 );
    // std::cout << "Old t: " << t.format( utils::eigenFormat() ) << std::endl;

    // PinholeCamera camera( 1242, 375, cameraMatrix, distortionCoeffs );
    // PinholeCamera camera( config->m_imgWidth, config->m_imgHeight, cameraMatrix, distortionCoeffs );
    // Frame refFrame( camera, refImg );
    system.processFirstFrame(refImg);
    // Frame curFrame( camera, curImg );

    // std::cout << "transformation W -> 1: " << refFrame.m_TransW2F.params().transpose() << std::endl;
    // std::cout << "transformation W -> 2: " << curFrame.m_TransW2F.params().transpose() << std::endl;
    // std::cout << "transformation 1 -> 2: " << Sophus::SE3d( temp.toRotationMatrix(), t ).params().transpose()
    //   << std::endl;
    // Sophus::SE3d T_pre2cur = refFrame.m_TransW2F.inverse() * curFrame.m_TransW2F;
    // std::cout << "transformation 1 -> 2: " << T_pre2cur.params().transpose() << std::endl;

    // const nlohmann::json& algoJson = configJson[ "algorithm" ];
    // const uint32_t numFeature       = algoJson[ "number_detected_features" ].get< uint32_t >();

    // keep it as int32_t. in detect fea
    // const uint32_t gridSize             = algoJson[ "grid_size_select_features" ].get< uint32_t >();
    // const uint32_t patchSizeOptFlow     = algoJson[ "patch_size_optical_flow" ].get< uint32_t >();
    // const uint32_t patchSize            = algoJson[ "patch_size_image_alignment" ].get< uint32_t >();
    // const uint32_t minLevelImagePyramid = algoJson[ "min_level_image_pyramid" ].get< uint32_t >();
    // const uint32_t maxLevelImagePyramid = algoJson[ "max_level_image_pyramid" ].get< uint32_t >();

    // FeatureSelection featureSelection( refFrame.m_imagePyramid.getBaseImage() );
    // std::cout << "Fundamental Matrix: \n" << F << std::endl;
    // Matcher matcher;

    auto t1 = std::chrono::high_resolution_clock::now();
    // featureSelection.detectFeaturesSSC( refFrame, numFeature );
    // featureSelection.detectFeaturesInGrid( refFrame, 20 );
    // std::cout << "# observation: " << refFrame.numberObservation() << std::endl;
    // visualization::featurePointsInGrid(featureSelection.m_gradientMagnitude, refFrame, patchSize,
    // "Feature-Point-In-Grid");

    // {
    //     cv::Mat refBGR = visualization::getBGRImage( refFrame.m_imagePyramid.getBaseImage() );
    //     visualization::featurePointsInGrid(refBGR, refFrame, patchSize );
    //     cv::imshow("grid_frame_0", refBGR);
    // }

    // Eigen::Matrix3d E;
    // Eigen::Matrix3d F;
    // Eigen::Matrix3d R;
    // Eigen::Vector3d t;

    // Matcher::computeOpticalFlowSparse( refFrame, curFrame, 11 );
    // Matcher::computeEssentialMatrix( refFrame, curFrame, 1.0, E );
    // Eigen::Matrix3d R2;
    // algorithm::decomposeEssentialMatrix( E, R, R2, t );
    // std::cout << "E: " << E.format( utils::eigenFormat() ) << std::endl;
    // std::cout << "R: " << R.format( utils::eigenFormat() ) << std::endl;
    // std::cout << "R2: " << R2.format( utils::eigenFormat() ) << std::endl;
    // std::cout << "t: " << t.format( utils::eigenFormat() ) << std::endl;
    // F = curFrame.m_camera->invK().transpose() * E * refFrame.m_camera->invK();
    // std::cout << "F: " << F.format( utils::eigenFormat() ) << std::endl;
    // algorithm::recoverPose( E, refFrame, curFrame, R, t );
    // std::cout << "R: " << R.format( utils::eigenFormat() ) << std::endl;
    // std::cout << "t: " << t.format( utils::eigenFormat() ) << std::endl;
    // std::cout << "E new: " << (R * algorithm::hat(t)).format( utils::eigenFormat() ) << std::endl;
    // std::cout << "determinant: " << R.determinant() << std::endl;
    // std::cout << "ref C: " << refFrame.cameraInWorld().format( utils::eigenFormat() ) << std::endl;
    // std::cout << "cur C: " << curFrame.cameraInWorld().format( utils::eigenFormat() ) << std::endl;
    // std::cout << "ref w-T: " << refFrame.m_TransW2F.translation().format( utils::eigenFormat() ) << std::endl;
    // std::cout << "cur w-T: " << curFrame.m_TransW2F.translation().format( utils::eigenFormat() ) << std::endl;

    // Eigen::AngleAxisd temp( R );  // Re-orthogonality
    // curFrame.m_TransW2F = refFrame.m_TransW2F * Sophus::SE3d( temp.toRotationMatrix(), t );
    // Eigen::MatrixXd pointsRefCamera(3, curFrame.numberObservation());
    // algorithm::pointsRefCamera(refFrame, curFrame, pointsRefCamera);
    // std::size_t numObserves = curFrame.numberObservation();
    // Eigen::MatrixXd pointsWorld( 3, numObserves );
    // Eigen::MatrixXd pointsCurCamera( 3, numObserves );
    // Eigen::VectorXd depthCurFrame( numObserves );
    // algorithm::points3DWorld( refFrame, curFrame, pointsWorld );
    // algorithm::transferPointsWorldToCam( curFrame, pointsWorld, pointsCurCamera );
    // algorithm::depthCamera( curFrame, pointsWorld, depthCurFrame );

    // for(int i(0); i< numObserves; i++)
    // {
    //     std::cout << "world pos: " << pointsWorld.col(i).transpose() << ", cur pos: " <<
    //     pointsCurCamera.col(i).transpose() << std::endl;
    // }
    // visualization::epipolarLinesWithDepth( refFrame, curFrame, depthCurFrame, 150.0, "Epipolar-Lines-Depths-first" );
    // visualization::epipolarLinesWithPoints( refFrame, curFrame, pointsWorld, 15.0, "Epipolar-Lines-Depths-first" );
    // algorithm::normalizedDepthsCurCamera( curFrame, pointsWorld, depthCurFrame );

    // double medianDepth = algorithm::computeMedian( depthCurFrame );
    // std::cout << "Mean: " << depthCurFrame.mean() << std::endl;
    // std::cout << "Median: " << medianDepth << std::endl;
    // const double scale = 1.0 / medianDepth;
    // std::cout << "translation without scale: " << curFrame.m_TransW2F.translation().transpose() << std::endl;

    // C_1 = C_0 + scale * (C_1 - C_0)
    // t = -R * C_1
    // curFrame.m_TransW2F.translation() = -curFrame.m_TransW2F.rotationMatrix() *
    //                                     ( refFrame.cameraInWorld() + scale * ( curFrame.cameraInWorld() - refFrame.cameraInWorld() ) );
    // std::cout << "translation with scale: " << curFrame.m_TransW2F.translation().transpose() << std::endl;

    // std::cout << "Before scaling depth 0: " << depthCurFrame(0) << std::endl;
    // depthCurFrame *= scale;
    // std::cout << "After scaling depth 0: " << depthCurFrame(0) << std::endl;

    // pointsCurCamera *= scale;
    // algorithm::transferPointsCamToWorld( curFrame, pointsCurCamera, pointsWorld );
    // uint32_t cnt = 0;
    // for ( std::size_t i( 0 ); i < numObserves; i++ )
    // {
    //     const Eigen::Vector2d refFeature = refFrame.m_frameFeatures[ i ]->m_feature;
    //     const Eigen::Vector2d curFeature = curFrame.m_frameFeatures[ i ]->m_feature;
    //     if ( refFrame.m_camera->isInFrame( refFeature, 5.0 ) == true && curFrame.m_camera->isInFrame( curFeature, 5.0 ) == true &&
    //          pointsCurCamera.col( i ).z() > 0 )
    //     {
    //         std::shared_ptr< Point > point = std::make_shared< Point >( pointsWorld.col( i ) );
    //         //    std::cout << "3D points: " << point->m_position.format(utils::eigenFormat()) << std::endl;
    //         refFrame.m_frameFeatures[ i ]->setPoint( point );
    //         curFrame.m_frameFeatures[ i ]->setPoint( point );
    //         cnt++;
    //     }
    // }
    // refFrame.m_frameFeatures.erase( std::remove_if( refFrame.m_frameFeatures.begin(), refFrame.m_frameFeatures.end(),
    //                                                 []( const auto& feature ) { return feature->m_point == nullptr; } ),
    //                                 refFrame.m_frameFeatures.end() );

    // curFrame.m_frameFeatures.erase( std::remove_if( curFrame.m_frameFeatures.begin(), curFrame.m_frameFeatures.end(),
    //                                                 []( const auto& feature ) { return feature->m_point == nullptr; } ),
    //                                 curFrame.m_frameFeatures.end() );

    // std::cout << "Points: " << pointsWorld.cols() << " cnt: " << cnt << " num ref observes: " << refFrame.numberObservation()
    //           << " num cur observes: " << curFrame.numberObservation() << std::endl;
    // algorithm::transferPointsCurToWorld(curFrame, pointsCurCamera, pointsWorld);

    // for ( std::size_t i( 0 ); i < numObserves; i++ )
    // {
    //     std::cout << "ref points 3D: " <<
    //     refFrame.m_frameFeatures[i]->m_point->m_position.format(utils::eigenFormat()) << std::endl; std::cout << "cur
    //     points 3D: " << curFrame.m_frameFeatures[i]->m_point->m_position.format(utils::eigenFormat()) << std::endl;
    // }
    // algorithm::normalizedDepthRefCamera(refFrame, curFrame, depthCurFrame);
    // R = R2;
    // Matcher::findTemplateMatch(refFrame, curFrame, patchSizeOptFlow, 35);

    system.processSecondFrame(curImg);
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "Elapsed time for matching: " << std::chrono::duration_cast< std::chrono::microseconds >( t2 - t1 ).count() << " micro sec"
              << std::endl;

/*
    numObserves = refFrame.numberObservation();
    Eigen::VectorXd newCurDepths( numObserves );
    algorithm::depthCamera( curFrame, newCurDepths );
    medianDepth           = algorithm::computeMedian( newCurDepths );
    const double minDepth = newCurDepths.minCoeff();
    // std::cout << "Mean: " << medianDepth << " min: " << minDepth << std::endl;
    curFrame.setKeyframe();
*/

    {
        // cv::Mat refBGR = visualization::getBGRImage( refFrame.m_imagePyramid.getBaseImage() );
        // cv::Mat curBGR = visualization::getBGRImage( curFrame.m_imagePyramid.getBaseImage() );
        // visualization::featurePointsInGrid(refBGR, refFrame, patchSize );
        // cv::imshow("grid_frame_0", refBGR);

        // visualization::featurePoints(refBGR, refFrame);
        // visualization::featurePoints( curBGR, curFrame );
        // visualization::project3DPoints(curBGR, curFrame);
        // visualization::projectPointsWithRelativePose( curBGR, refFrame, curFrame );
        // cv::Mat stickImg;
        // visualization::stickTwoImageHorizontally(refBGR, curBGR, stickImg);
        // cv::imshow("both image", stickImg);
        // cv::imshow("relative_0_1", curBGR);
    }


    const cv::Mat newImg = cv::imread( utils::findAbsoluteFilePath( "input/0000000002.png" ), cv::IMREAD_GRAYSCALE );
    // Frame newFrame( camera, newImg );
    system.processFrame(newImg);
    system.reportSummaryFrames();
    system.reportSummaryFeatures();
    // {
    //     cv::Mat curBGR = visualization::getBGRImage( curFrame.m_imagePyramid.getBaseImage() );
    //     cv::Mat newBGR = visualization::getBGRImage( newFrame.m_imagePyramid.getBaseImage() );
    //     visualization::featurePoints( curBGR, curFrame);
    //     // visualization::featurePoints(newBGR, newFrame);
    //     // visualization::project3DPoints(curBGR, curFrame);
    //     visualization::projectPointsWithRelativePose( newBGR, curFrame, newFrame );
    //     cv::Mat stickImg;
    //     visualization::stickTwoImageHorizontally( curBGR, newBGR, stickImg );
    //     // cv::imshow("both_image_1_2", stickImg);
    //     // cv::imshow("relative_1_2", newBGR);
    // }

/*
    ImageAlignment match( 5, 0, 3, 6 );
    t1 = std::chrono::high_resolution_clock::now();
    match.align( curFrame, newFrame );
    t2 = std::chrono::high_resolution_clock::now();
    std::cout << "Elapsed time for alignment: " << std::chrono::duration_cast< std::chrono::microseconds >( t2 - t1 ).count() << " micro sec"
              << std::endl;

    {
        cv::Mat curBGR = visualization::getBGRImage( curFrame.m_imagePyramid.getBaseImage() );
        cv::Mat newBGR = visualization::getBGRImage( newFrame.m_imagePyramid.getBaseImage() );
        visualization::featurePoints( curBGR, curFrame );
        // visualization::featurePointsInGrid(curBGR, curFrame, 50);
        // visualization::featurePoints(newBGR, newFrame);
        // visualization::project3DPoints(curBGR, curFrame);
        visualization::projectPointsWithRelativePose( newBGR, curFrame, newFrame );
        cv::Mat stickImg;
        visualization::stickTwoImageHorizontally( curBGR, newBGR, stickImg );
        // cv::imshow( "both_image_1_2_optimization", stickImg );
        // cv::imshow("relative_1_2", newBGR);
    }
*/

    // matcher.findTemplateMatch(refFrame, curFrame, 5, 99);
    // visualization visualize;
    // Eigen::MatrixXd P1 = refFrame.m_camera->K() * refFrame.m_TransW2F.matrix3x4();
    // std::cout << "P1: " << P1 << std::endl;
    // Eigen::MatrixXd P2 = curFrame.m_camera->K() * curFrame.m_TransW2F.matrix3x4();
    // std::cout << "P2: " << P2 << std::endl;

    // Triangulation triangulate;
    // Eigen::Vector3d point;
    // Eigen::Vector2d p1( 975, 123 );
    // Eigen::Vector2d p2( 1004, 119 );
    // algorithm::triangulatePointHomogenousDLT( refFrame, curFrame, p1, p2, point );
    // algorithm::triangulatePointDLT( refFrame, curFrame, p1, p2, point );
    // std::cout << "point in world: " << point.transpose() << std::endl;
    // std::cout << "point: " << point.norm() << std::endl;
    // C = curFrame.cameraInWorld();
    // std::cout << "C: " << C.format( utils::eigenFormat() ) << std::endl;

    // visualization::featurePoints( featureSelection.m_gradientMagnitude, refFrame,
    //   "Feature Selected By SSC on Gradient Magnitude Image" );
    // visualization::epipole( curFrame, C, "Epipole-Right" );
    // visualization::epipolarLine(img, vecHomo, K, R, t, "Epipolar-Line");
    // visualization::epipolarLine( curFrame, refFrame.m_frameFeatures[ 0 ]->m_bearingVec, 0.0, 50.0,
    //  "Epipolar-Line-Feature-0" );
    // const double mu    = point.norm();
    // std::cout << "position feature: " << refFrame.m_frameFeatures[ 3 ]->m_feature.transpose() << std::endl;
    // {
    // const double mu    = 20.0;
    // const double sigma = 1.0;
    // visualization::epipolarLine( refFrame, curFrame, refFrame.m_frameFeatures[ 3 ]->m_feature, mu - sigma,
    //                                 mu + sigma, "Epipolar-Line-Feature-3" );
    // }

    // visualization::epipolarLinesWithPoints( refFrame, curFrame, pointsWorld, 150.0, "Epipolar-Lines-Depths" );
    // visualization::featurePointsInGrid(featureSelection.m_gradientMagnitude, refFrame, patchSize,
    // "Feature-Point-In-Grid");

    // visualization::epipolarLine( refFrame, curFrame, refFrame.m_frameFeatures[ 3 ]->m_feature, 0.5, 20,
    //  "Epipolar-Line-Feature-3" );

    // visualization::featurePointsInBothImages( refFrame, curFrame, "Feature in Both Images" );
    // visualization::featurePointsInBothImagesWithSearchRegion( refFrame, curFrame, patchSizeOptFlow,
    // "Feature in Both Images" );

    // visualization::epipolarLinesWithFundamentalMatrix( refFrame, curFrame.m_imagePyramid.getBaseImage(), F,
    // "Epipolar-Lines-Right-With-F" );
    // visualization::epipolarLinesWithEssentialMatrix( refFrame, curFrame.m_imagePyramid.getBaseImage(), E,
    //  "Epipolar-Lines-Right-With-E" );
    // visualization::epipolarLinesWithPointsWithFundamentalMatrix( refFrame, curFrame, F,
    //  "Epipolar-Lines-with-Points-in-Right" );
    // visualization::grayImage( featureSelection.m_gradientMagnitude, "Gradient Magnitude" );
    // visualization::HSVColoredImage( featureSelection.m_gradientMagnitude, "Gradient Magnitude HSV" );
    // visualization::HSVColoredImage( featureSelection.m_gradientMagnitude, "Gradient Magnitude HSV" );

    cv::waitKey( 0 );
    return 0;
}
