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

#include <Eigen/Core>
#include <iostream>
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
    std::pair< std::uint16_t, std::uint16_t > m_imgRes;
    Sophus::SE3d m_TransW2F;
    Eigen::Matrix< double, 6, 6 > m_covPose;
    ImagePyramid m_imagePyramid;
    std::vector< Feature* > m_frameFeatures;
    bool m_keyFrame;

    // C'tor
    explicit Frame( Eigen::Matrix3d& K, cv::Mat& img );
    // Copy C'tor
    Frame( const Frame& rhs ) = delete;  // non construction-copyable
    // move C'tor
    Frame( Frame&& rhs ) = delete;  // non copyable
    // Copy assignment operator
    Frame& operator=( const Frame& rhs ) = delete;  // non construction movable
    // move assignment operator
    Frame& operator=( Frame&& rhs ) = delete;  // non movable
    // D'tor
    ~Frame() = default;

    /// Initialize new frame and create image pyramid.
    void initFrame( const cv::Mat& img );

    /// Select this frame as keyframe.
    void setKeyframe();

    /// Add a feature to the image
    void addFeature( Feature* feature );

    void removeKeyPoint( Feature* feature );

    std::uint32_t numberObservation() const;

    bool isVisible( const Eigen::Vector3d& point3D ) const;

    bool isKeyframe() const;

    /// project from world to image pixel coordinate
    Eigen::Vector2d world2image( const Eigen::Vector3d& point3D_w ) const;

    /// project from world to camera coordinate
    Eigen::Vector3d world2camera( const Eigen::Vector3d& point3D_w ) const;

    /// project from camera coordinate to world
    Eigen::Vector3d camera2world( const Eigen::Vector3d& point3D_c ) const;

    /// project from camera coordinate to image pixel coordinate
    Eigen::Vector2d camera2image( const Eigen::Vector3d& point3D_c ) const;

    /// project from image pixel coordinate to world cooridnate
    Eigen::Vector3d image2world( const Eigen::Vector2d& point2D, const double depth ) const;

    /// project from image pixel cooridnate to camera coordinate
    Eigen::Vector3d image2camera( const Eigen::Vector2d& point2D, const double depth ) const;

    /// compute position of camera in world cordinate C = -R^t * t
    Eigen::Vector3d cameraInWorld() const;

private:
};

#endif /* __FRAME_H__ */