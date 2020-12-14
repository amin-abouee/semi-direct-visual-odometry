#include "visualization.hpp"

#include "gtest/gtest.h"
#include <opencv2/highgui.hpp>

#if 0
TEST(Visualization, ImageGrid)
{
    cv::Mat input(500, 500, CV_8UC3);
    cv::randu(input, cv::Scalar(0, 0, 0), cv::Scalar(50, 50, 50));
    cv::imshow("input", input);

    cv::Mat result = visualization::imageGrid(input, 22, 100);
    cv::imshow("result", result);

    cv::waitKey(0);
}
#endif

#if 0
TEST(Visualization, GetImage)
{
    cv::Mat input(500, 500, CV_8UC3);

    cv::randu(input, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));
    cv::imshow("input", input);

    cv::Mat resGray = visualization::getGrayImage(input);
    cv::imshow("Gray", resGray);

    cv::Mat resHSV = visualization::getHSVImage(input);
    cv::imshow("HSV", resHSV);

    cv::waitKey(0);
}
#endif

#if 0
TEST(Visualization, StickImages)
{
    cv::Mat input1(500, 500, CV_8UC3);
    cv::Mat input2(500, 500, CV_8UC3);

    cv::randu(input1, cv::Scalar(0, 0, 0), cv::Scalar(100, 100, 100));
    cv::randu(input2, cv::Scalar(100, 100, 100), cv::Scalar(255, 255, 255));

    cv::imshow("input1", input1);
    cv::imshow("input2", input2);

    cv::Mat resVertical = visualization::stickImagesVertically(input1, input2);
    cv::imshow("resVertical", resVertical);

    cv::Mat resHorizontal = visualization::stickImagesHorizontally(input1, input2);
    cv::imshow("resHorizontal", resHorizontal);

    cv::waitKey(0);
}
#endif
