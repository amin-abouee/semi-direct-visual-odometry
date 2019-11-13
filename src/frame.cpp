#include "frame.hpp"

Frame::Frame(Eigen::Matrix3d& K, cv::Mat& img): m_imagePyramid(img, 4)
{
    
}