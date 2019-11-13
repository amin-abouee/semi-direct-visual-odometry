/**
* @file Frame.hpp
* @brief frame information
*
* @date 18.11.2019
* @author Amin Abouee
*
* @section DESCRIPTION
*
*
*/
#ifndef __FRAME_HPP__
#define __FRAME_HPP__

#include <iostream>
#include <Eigen/Core>
#include <sophus/se3.hpp>

#include "image_pyramid.hpp"

class Feature;
class Point;

class Frame final
{
public:

    static uint32_t m_frameCounter;
    uint32_t m_id;
    Eigen::Matrix3d m_K;
    Sophus::SE3d m_TransformWF;
    Eigen::Matrix<double, 6, 6> m_covPose;
    ImagePyramid m_imagePyramid;
    std::vector <Feature*> m_currentFrameFeatures;
    bool m_keyFrame;


    //C'tor
    explicit Frame(Eigen::Matrix3d& K, cv::Mat& img);
    //Copy C'tor
    Frame(const Frame& rhs) = delete; // non construction-copyable
    //move C'tor
    Frame(Frame&& rhs) = delete; // non copyable
    //Copy assignment operator
    Frame& operator=(const Frame& rhs) = delete; // non construction movable
    //move assignment operator
    Frame& operator=(Frame&& rhs) = delete; // non movable
    //D'tor
    ~Frame() = default;

 private:

 };

 #endif /* __FRAME_H__ */