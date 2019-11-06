#include <algorithm>
#include <chrono>
#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <Eigen/Core>

#include "feature-selection.hpp"
#include "visualization.hpp"

int main( int argc, char* argv[] )
{
    cv::Mat img = cv::imread( "../input/0000000000.png", cv::IMREAD_GRAYSCALE );

    Eigen::Matrix3d K;
    K << 7.215377e+02, 0.000000e+00, 6.095593e+02, 0.000000e+00, 7.215377e+02, 1.728540e+02, 0.000000e+00,
      0.000000e+00, 1.000000e+00;
    std::cout << "Camera Matrix: " << K << std::endl;

    Eigen::Matrix3d E;
    E << .22644456e-03, -7.06943058e-01, -4.05822481e-03, 7.06984545e-01, 1.22048201e-03, 1.26855863e-02,
      3.25653616e-03, -1.46073125e-02, -2.59077801e-05;
    std::cout << "Essential Matrix: " << E << std::endl;

    Eigen::Matrix3d F;
    F << -5.33286713e-08, -1.49632194e-03,  2.67961447e-01, 1.49436356e-03, -2.27291565e-06, -9.03327631e-01,
      -2.68937438e-01,  9.02739500e-01,  1.00000000e+00;
    std::cout << "Fundamental Matrix: " << E << std::endl;

    FeatureSelection featureSelection( img );

    // auto t1 = std::chrono::high_resolution_clock::now();
    // featureSelection.detectFeatures();
    // auto t2 = std::chrono::high_resolution_clock::now();
    // std::cout << "Elapsed time for gradient magnitude: " <<
    // std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count() << std::endl;

    // double min, max;
    // cv::minMaxLoc(mag, &min, &max);

    // cv::Mat histoMag;
    // float range[] = { 0, 250 }; //the upper boundary is exclusive
    // const float* histRange = { range };
    // int histSize = 10;
    // cv::calcHist(&mag, 1, 0, cv::Mat(), histoMag, 1, &histSize, &histRange);

    auto t3 = std::chrono::high_resolution_clock::now();
    featureSelection.detectFeatures( 500 );
    auto t4 = std::chrono::high_resolution_clock::now();
    ;
    std::cout << "Elapsed time for SSC: " << std::chrono::duration_cast< std::chrono::milliseconds >( t4 - t3 ).count()
              << std::endl;

    // featureSelection.visualizeFeaturePoints();
    // featureSelection.visualizeGrayGradientMagnitude();
    // featureSelection.visualizeColoredGradientMagnitude();
    const Eigen::Vector3d C( -0.01793327, -0.00577164, 1 );
    
    Eigen::Matrix3d R;
    R << 0.99999475,  0.0017505,  -0.0027263, -0.00174731,  0.99999779,  0.00117013, 0.00272834, -0.00116536,  0.9999956;

    Eigen::Vector3d t(-0.0206659, -0.00456935, 0.999776);

    // Eigen::Vector3d vec = featureSelection.m_kp.col(0);
    // Eigen::Vector3d vecHomo(vec(0), vec(1), 1.0);

    Visualization visualize;
    visualize.visualizeEpipole( img, C, K, "Epipole-Left" );
    // visualize.visualizeEpipolarLine(img, vecHomo, K, R, t, "Epipolar-Line");
    // visualize.visualizeEpipolarLine(img, vec, E, "Epipolar-Line");
    visualize.visualizeEpipolarLines(img, featureSelection.m_kp, F, "Epipolar-Lines-Right");
    visualize.visualizeFeaturePoints(featureSelection.m_gradientMagnitude, featureSelection.m_kp, "Selected By SSC");
    visualize.visualizeGrayImage(featureSelection.m_gradientMagnitude, "Gradient Magnitude");
    visualize.visualizeHSVColoredImage(featureSelection.m_gradientMagnitude, "Gradient Magnitude HSV");
    cv::waitKey( 0 );
    return 0;
}
