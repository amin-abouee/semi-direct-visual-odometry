#include <iostream>
#include <algorithm>
#include <chrono>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <Eigen/Core>

#include "feature-selection.hpp"

int main(int argc, char* argv[])
{
    cv::Mat img = cv::imread("../input/0000000000.png", cv::IMREAD_GRAYSCALE);

    FeatureSelection featureSelection(img);

    // auto t1 = std::chrono::high_resolution_clock::now();
    // featureSelection.detectFeatures();
    // auto t2 = std::chrono::high_resolution_clock::now();
    // std::cout << "Elapsed time for gradient magnitude: " << std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count() << std::endl;

    // double min, max;
    // cv::minMaxLoc(mag, &min, &max);
    
    // cv::Mat histoMag;
    // float range[] = { 0, 250 }; //the upper boundary is exclusive
    // const float* histRange = { range };
    // int histSize = 10;
    // cv::calcHist(&mag, 1, 0, cv::Mat(), histoMag, 1, &histSize, &histRange);


    auto t3 = std::chrono::high_resolution_clock::now();
    featureSelection.detectFeatures(1000);
    auto t4 = std::chrono::high_resolution_clock::now();
    ;std::cout << "Elapsed time for SSC: " << std::chrono::duration_cast<std::chrono::milliseconds>(t4-t3).count() << std::endl;

    featureSelection.visualizeFeaturePoints();
    featureSelection.visualizeGrayGradientMagnitude();
    featureSelection.visualizeColoredGradientMagnitude();
    cv::waitKey(0);
    return 0;
}
