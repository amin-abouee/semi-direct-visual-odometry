#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <Eigen/Core>

#include "feature_selection.hpp"
#include "frame.hpp"
#include "pinhole_camera.hpp"
#include "visualization.hpp"

#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/spdlog.h"

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

int main( int argc, char* argv[] )
{
    auto mainLogger = spdlog::stdout_color_mt( "main" );
    mainLogger->set_level( spdlog::level::debug );
    mainLogger->set_pattern( "[%Y-%m-%d %H:%M:%S] [%s:%#] [%n->%l] [thread:%t] %v" );
    mainLogger->info( "Info" );
    mainLogger->debug( "Debug" );
    mainLogger->warn( "Warn" );
    mainLogger->error( "Error" );

    std::string configIOFile = "../config/config.json";

    const nlohmann::json& configJson = createConfigParser( configIOFile );
    // std::cout << configJson[ "file_paths" ][ "camera_calibration" ].get< std::string >() << std::endl;
    // std::ifstream jsonFile(configFile);
    // const nlohmann::json& filePaths = configFile[ "file_paths" ];

    cv::Mat refImg = cv::imread( "../input/0000000000.png", cv::IMREAD_GRAYSCALE );
    cv::Mat curImg = cv::imread( "../input/0000000001.png", cv::IMREAD_GRAYSCALE );

    Eigen::Matrix3d K;
    K << 7.215377e+02, 0.000000e+00, 6.095593e+02, 0.000000e+00, 7.215377e+02, 1.728540e+02, 0.000000e+00, 0.000000e+00,
      1.000000e+00;
    std::cout << "Camera Matrix: \n" << K << std::endl;

    Eigen::Matrix3d E;
    E << .22644456e-03, -7.06943058e-01, -4.05822481e-03, 7.06984545e-01, 1.22048201e-03, 1.26855863e-02,
      3.25653616e-03, -1.46073125e-02, -2.59077801e-05;
    std::cout << "Essential Matrix: \n" << E << std::endl;

    Eigen::Matrix3d F;
    F << -5.33286713e-08, -1.49632194e-03, 2.67961447e-01, 1.49436356e-03, -2.27291565e-06, -9.03327631e-01,
      -2.68937438e-01, 9.02739500e-01, 1.00000000e+00;
    std::cout << "Fundamental Matrix: \n" << F << std::endl;

    const Eigen::Vector3d C( -0.01793327, -0.00577164, 1 );

    Eigen::Matrix3d R;
    R << 0.99999475, 0.0017505, -0.0027263, -0.00174731, 0.99999779, 0.00117013, 0.00272834, -0.00116536, 0.9999956;

    Eigen::Vector3d t( -0.0206659, -0.00456935, 0.999776 );

    PinholeCamera camera( 1242, 375, K( 0, 0 ), K( 1, 1 ), K( 0, 2 ), K( 1, 2 ), 0.0, 0.0, 0.0, 0.0, 0.0 );
    Frame refFrame( camera, refImg );
    Frame curFrame( camera, curImg );
    Eigen::AngleAxisd temp( R );  // Re-orthogonality
    curFrame.m_TransW2F = refFrame.m_TransW2F * Sophus::SE3d( temp.toRotationMatrix(), t );
    // std::cout << "transformation W -> 1: " << refFrame.m_TransW2F.params().transpose() << std::endl;
    // std::cout << "transformation W -> 2: " << curFrame.m_TransW2F.params().transpose() << std::endl;
    // std::cout << "transformation 1 -> 2: " << Sophus::SE3d( temp.toRotationMatrix(), t ).params().transpose()
            //   << std::endl;
    Sophus::SE3d T_pre2cur = refFrame.m_TransW2F.inverse() * curFrame.m_TransW2F;
    std::cout << "transformation 1 -> 2: " << T_pre2cur.params().transpose() << std::endl;

    FeatureSelection featureSelection;
    F = curFrame.m_camera->invK().transpose() * E * refFrame.m_camera->invK();
    std::cout << "Fundamental Matrix: \n" << F << std::endl;

    auto t1 = std::chrono::high_resolution_clock::now();
    featureSelection.detectFeatures( refFrame, 5 );
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "Elapsed time for SSC: " << std::chrono::duration_cast< std::chrono::milliseconds >( t2 - t1 ).count()
              << std::endl;

    Visualization visualize;
    // visualize.visualizeFeaturePoints( featureSelection.m_gradientMagnitude, refFrame,
    //   "Feature Selected By SSC on Gradient Magnitude Image" );
    visualize.visualizeEpipole( curFrame, C, "Epipole-Right" );
    // visualize.visualizeEpipolarLine(img, vecHomo, K, R, t, "Epipolar-Line");
    // visualize.visualizeEpipolarLine( curFrame, refFrame.m_frameFeatures[ 0 ]->m_bearingVec, 0.0, 50.0,
    //                                  "Epipolar-Line-Feature-0" );
    visualize.visualizeEpipolarLine( refFrame, curFrame, refFrame.m_frameFeatures[ 3 ]->m_feature, 0.0, 1e12,
                                     "Epipolar-Line-Feature-0" );
    visualize.visualizeEpipolarLinesWithFundamenalMatrix( refFrame, curFrame.m_imagePyramid.getBaseImage(), F,
                                                          "Epipolar-Lines-Right-With-F" );
    visualize.visualizeEpipolarLinesWithEssentialMatrix( refFrame, curFrame.m_imagePyramid.getBaseImage(), E,
                                                          "Epipolar-Lines-Right-With-E" );
    // visualize.visualizeGrayImage( featureSelection.m_gradientMagnitude, "Gradient Magnitude" );
    // visualize.visualizeHSVColoredImage( featureSelection.m_gradientMagnitude, "Gradient Magnitude HSV" );
    // visualize.visualizeHSVColoredImage( featureSelection.m_gradientMagnitude, "Gradient Magnitude HSV" );
    cv::waitKey( 0 );
    return 0;
}
