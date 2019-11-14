#include "frame.hpp"

Frame::Frame(Eigen::Matrix3d& K, cv::Mat& img): m_imagePyramid(img, 4)
{
    
}

void Frame::initFrame(const cv::Mat& img)
{
    if (img.empty())
        throw std::runtime_error("BANGO");
}