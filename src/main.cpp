#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
// #include <opencv2/core/eigen.hpp>

#include <Eigen/Core>

#include "algorithm.hpp"
#include "feature_selection.hpp"
#include "frame.hpp"
#include "matcher.hpp"
#include "pinhole_camera.hpp"
#include "visualization.hpp"

// #include "spdlog/sinks/stdout_color_sinks.h"
#include <spdlog/spdlog.h>

#include <nlohmann/json.hpp>

nlohmann::json createConfigParser( const std::string& fileName )
{
    nlohmann::json configParser;
    try
    {
        std::ifstream fileReader( fileName );
        configParser = nlohmann::json::parse( fileReader );
        fileReader.close();
    }
    catch ( const std::exception& e )
    {
        std::cerr << "JSON Parser Error:" << e.what() << '\n';
    }
    return configParser;
}

bool loadCameraIntrinsics( const std::string& filename, cv::Mat& cameraMatrix, cv::Mat& distortionCoeffs )
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
    catch (std::exception& e)
    {
        std::cout << e.what() << std::endl;
        return false;
    }
}

int main( int argc, char* argv[] )
{
    Eigen::IOFormat CommaInitFmt( 6, Eigen::DontAlignCols, ", ", ", ", "", "", " [ ", "]" );
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
    if (argc > 1)
        configIOFile = argv[1];
    else
        configIOFile = "../config/config.json";
    

    const nlohmann::json& configJson = createConfigParser( configIOFile );
    // std::cout << configJson[ "file_paths" ][ "camera_calibration" ].get< std::string >() << std::endl;
    // std::ifstream jsonFile(configFile);

    const nlohmann::json& cameraJson  = configJson[ "camera" ];
    const std::string calibrationFile = cameraJson[ "camera_calibration" ].get< std::string >();
    cv::Mat cameraMatrix;
    cv::Mat distortionCoeffs;
    bool result = loadCameraIntrinsics( calibrationFile, cameraMatrix, distortionCoeffs );
    if (result == false)
    {
        std::cout << "Failed to open the calibration file, check config.json file" << std::endl;
        return EXIT_FAILURE;
    }

    const int32_t imgWidth  = cameraJson[ "img_width" ].get< int32_t >();
    const int32_t imgHeight = cameraJson[ "img_height" ].get< int32_t >();

    const cv::Mat refImg = cv::imread( "../input/0000000000.png", cv::IMREAD_GRAYSCALE );
    const cv::Mat curImg = cv::imread( "../input/0000000001.png", cv::IMREAD_GRAYSCALE );

    Eigen::Matrix3d K;
    K << 7.215377e+02, 0.000000e+00, 6.095593e+02, 0.000000e+00, 7.215377e+02, 1.728540e+02, 0.000000e+00, 0.000000e+00,
      1.000000e+00;
    // std::cout << "Camera Matrix: \n" << K << std::endl;

    Eigen::Matrix3d E;
    E << .22644456e-03, -7.06943058e-01, -4.05822481e-03, 7.06984545e-01, 1.22048201e-03, 1.26855863e-02,
      3.25653616e-03, -1.46073125e-02, -2.59077801e-05;
    std::cout << "Old E: " << E.format( CommaInitFmt ) << std::endl;

    Eigen::Matrix3d F;
    F << -5.33286713e-08, -1.49632194e-03, 2.67961447e-01, 1.49436356e-03, -2.27291565e-06, -9.03327631e-01,
      -2.68937438e-01, 9.02739500e-01, 1.00000000e+00;
    // std::cout << "Fundamental Matrix: \n" << F << std::endl;

    Eigen::Vector3d C( -0.01793327, -0.00577164, 1 );

    // Do the cheirality check.
    Eigen::Matrix3d R;
    R << 0.99999475, 0.0017505, -0.0027263, -0.00174731, 0.99999779, 0.00117013, 0.00272834, -0.00116536, 0.9999956;
    std::cout << "Old R1: " << R.format( CommaInitFmt ) << std::endl;

    // Eigen::Matrix3d R;
    // R << -0.99925367, -0.00151199, -0.03859818, 0.00191117, -0.99994505, -0.01030721, -0.03858047, -0.01037328,
    // 0.99920165;

    // Eigen::Vector3d t( -0.0206659, -0.00456935, 0.999776 );
    Eigen::Vector3d t( 0.0206659, 0.00456935, -0.999776 );
    std::cout << "Old t: " << t.format( CommaInitFmt ) << std::endl;

    // PinholeCamera camera( 1242, 375, K( 0, 0 ), K( 1, 1 ), K( 0, 2 ), K( 1, 2 ), 0.0, 0.0, 0.0, 0.0, 0.0 );
    PinholeCamera camera( imgWidth, imgHeight, cameraMatrix, distortionCoeffs );
    Frame refFrame( camera, refImg );
    Frame curFrame( camera, curImg );

    // std::cout << "transformation W -> 1: " << refFrame.m_TransW2F.params().transpose() << std::endl;
    // std::cout << "transformation W -> 2: " << curFrame.m_TransW2F.params().transpose() << std::endl;
    // std::cout << "transformation 1 -> 2: " << Sophus::SE3d( temp.toRotationMatrix(), t ).params().transpose()
    //   << std::endl;
    // Sophus::SE3d T_pre2cur = refFrame.m_TransW2F.inverse() * curFrame.m_TransW2F;
    // std::cout << "transformation 1 -> 2: " << T_pre2cur.params().transpose() << std::endl;

    const nlohmann::json& algoJson  = configJson[ "algorithm" ];
    const uint32_t numFeature       = algoJson[ "number_detected_features" ].get< uint32_t >();
    const uint32_t patchSize       = algoJson[ "grid_size_select_features" ].get< uint32_t >();
    const uint16_t patchSizeOptFlow = algoJson[ "patch_size_optical_flow" ].get< uint16_t >();

    FeatureSelection featureSelection(refFrame.m_imagePyramid.getBaseImage());
    // std::cout << "Fundamental Matrix: \n" << F << std::endl;
    // Matcher matcher;

    auto t1 = std::chrono::high_resolution_clock::now();
    // featureSelection.detectFeaturesSSC( refFrame, numFeature );
    featureSelection.detectFeaturesInGrid( refFrame, patchSize );
    // Visualization::featurePointsInGrid(featureSelection.m_gradientMagnitude, refFrame, patchSize, "Feature-Point-In-Grid");

    Matcher::computeOpticalFlowSparse( refFrame, curFrame, patchSizeOptFlow );
    Matcher::computeEssentialMatrix( refFrame, curFrame, 1.0, E );
    Eigen::Matrix3d R2;
    Algorithm::decomposeEssentialMatrix( E, R, R2, t );
    std::cout << "E: " << E.format( CommaInitFmt ) << std::endl;
    std::cout << "R1: " << R.format( CommaInitFmt ) << std::endl;
    std::cout << "R2: " << R2.format( CommaInitFmt ) << std::endl;
    std::cout << "t: " << t.format( CommaInitFmt ) << std::endl;
    F = curFrame.m_camera->invK().transpose() * E * refFrame.m_camera->invK();
    Eigen::AngleAxisd temp( R );  // Re-orthogonality
    curFrame.m_TransW2F = refFrame.m_TransW2F * Sophus::SE3d( temp.toRotationMatrix(), t );
    // Eigen::MatrixXd pointsRefCamera(3, curFrame.numberObservation());
    // Algorithm::pointsRefCamera(refFrame, curFrame, pointsRefCamera);
    Eigen::VectorXd depthCurFrame(curFrame.numberObservation());
    Algorithm::normalizedDepthRefCamera(refFrame, curFrame, depthCurFrame);
    // R = R2;
    // Matcher::findTemplateMatch(refFrame, curFrame, patchSizeOptFlow, 35);
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "Elapsed time for SSC: " << std::chrono::duration_cast< std::chrono::milliseconds >( t2 - t1 ).count()
              << " ms" << std::endl;

    // matcher.findTemplateMatch(refFrame, curFrame, 5, 99);
    // Visualization visualize;
    // Eigen::MatrixXd P1 = refFrame.m_camera->K() * refFrame.m_TransW2F.matrix3x4();
    // std::cout << "P1: " << P1 << std::endl;
    // Eigen::MatrixXd P2 = curFrame.m_camera->K() * curFrame.m_TransW2F.matrix3x4();
    // std::cout << "P2: " << P2 << std::endl;

    // Triangulation triangulate;
    // Eigen::Vector3d point;
    // Eigen::Vector2d p1( 975, 123 );
    // Eigen::Vector2d p2( 1004, 119 );
    // Algorithm::triangulatePointHomogenousDLT( refFrame, curFrame, p1, p2, point );
    // Algorithm::triangulatePointDLT( refFrame, curFrame, p1, p2, point );
    // std::cout << "point in world: " << point.transpose() << std::endl;
    // std::cout << "point: " << point.norm() << std::endl;
    C = curFrame.cameraInWorld();
    std::cout << "C: " << C.format( CommaInitFmt ) << std::endl;

    // Visualization::featurePoints( featureSelection.m_gradientMagnitude, refFrame,
    //   "Feature Selected By SSC on Gradient Magnitude Image" );
    // Visualization::epipole( curFrame, C, "Epipole-Right" );
    // Visualization::epipolarLine(img, vecHomo, K, R, t, "Epipolar-Line");
    // Visualization::epipolarLine( curFrame, refFrame.m_frameFeatures[ 0 ]->m_bearingVec, 0.0, 50.0,
    //  "Epipolar-Line-Feature-0" );
    // const double mu    = point.norm();
    // std::cout << "position feature: " << refFrame.m_frameFeatures[ 3 ]->m_feature.transpose() << std::endl;
    {
        // const double mu    = 20.0;
        // const double sigma = 1.0;
        // Visualization::epipolarLine( refFrame, curFrame, refFrame.m_frameFeatures[ 3 ]->m_feature, mu - sigma,
        //                                 mu + sigma, "Epipolar-Line-Feature-3" );
    }

    Visualization::epipolarLinesWithDepth(refFrame, curFrame, depthCurFrame, 2.0, "Epipolar-Lines-Depths");
    Visualization::featurePointsInGrid(featureSelection.m_gradientMagnitude, refFrame, patchSize, "Feature-Point-In-Grid");

    // Visualization::epipolarLine( refFrame, curFrame, refFrame.m_frameFeatures[ 3 ]->m_feature, 0.5, 20,
                                //  "Epipolar-Line-Feature-3" );

    // Visualization::featurePointsInBothImages( refFrame, curFrame, "Feature in Both Images" );
    // Visualization::featurePointsInBothImagesWithSearchRegion( refFrame, curFrame, patchSizeOptFlow,
                                                              // "Feature in Both Images" );

    // Visualization::epipolarLinesWithFundamentalMatrix( refFrame, curFrame.m_imagePyramid.getBaseImage(), F,
    // "Epipolar-Lines-Right-With-F" );
    // Visualization::epipolarLinesWithEssentialMatrix( refFrame, curFrame.m_imagePyramid.getBaseImage(), E,
    //  "Epipolar-Lines-Right-With-E" );
    Visualization::epipolarLinesWithPointsWithFundamentalMatrix( refFrame, curFrame, F,
                                                                 "Epipolar-Lines-with-Points-in-Right" );
    // Visualization::grayImage( featureSelection.m_gradientMagnitude, "Gradient Magnitude" );
    // Visualization::HSVColoredImage( featureSelection.m_gradientMagnitude, "Gradient Magnitude HSV" );
    // Visualization::HSVColoredImage( featureSelection.m_gradientMagnitude, "Gradient Magnitude HSV" );
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
    // std::cout << "Number of Threads: " << Eigen::nbThreads( ) << std::endl;

    cv::waitKey( 0 );
    return 0;
}
