/**
 * @file Frame.hpp
 * @brief frame information
 *
 * @date 18.11.2019
 * @author Amin Abouee
 *
 * @section DESCRIPTION
 *
 */
#ifndef FRAME_HPP
#define FRAME_HPP

#include "image_pyramid.hpp"
#include "pinhole_camera.hpp"

#include <Eigen/Core>
#include <sophus/se3.hpp>
#include <g2o/types/sba/vertex_se3_expmap.h>

#include <iostream>
#include <memory>

/// @brief Feature information, is tracked across frames.
class Feature;
/**
 * @brief This class contains the RGB image informations, the absolute pose and its covariance and the camera geometry data
 *
 *
 *              Z
                ^
                |
                | World
                |
                O------>X
               / ^                                            ^
              /   \                                          /
             v    |                                         /
            Y      \                                       /
                    \                                     /
                     \                                   /
                     |                   u              /
                      \               +----->---------------------+
                       \              | Image                     |
                       |            v | Pixel                     |
                        \             v             x (cx, cy)    |
                       t \            |            /              |
                          \           |           /               |
                          |           |          /                |
                           \          +---------/-----------------+
    Trans W to C = [R|t]    \                  /
                             \                /
                             |               /
                              \             /
                               \           /
                               |          /
                                \        /
                                 \    Z /
                                  \    /
                                  |   /
                                   \ /
                                    O-------->X
                                    |
                                    | Camera
                                    |
                                    v
                                    Y
 *
 */
class Frame final
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    /// @brief Construct a new Frame object
    ///
    /// @param[in] camera Camera model
    /// @param[in] img Image data as opencv format
    /// @param[in] maxImagePyramid Maximum level of image pyramid. The image pyramid defines between level 0 (finest) and level
    /// maxImagePyramid (coarsest)
    /// @param[in] timestamp Timestamp of input image
    explicit Frame( const std::shared_ptr< PinholeCamera >& camera,
                    const cv::Mat& img,
                    const uint32_t maxImagePyramid,
                    const uint64_t timestamp,
                    const std::shared_ptr <Frame> lastKeyframe );
    // Copy C'tor
    Frame( const Frame& rhs ) = delete;  // non construction-copyable
    // move C'tor
    Frame( Frame&& rhs ) = delete;  // non construction movable
    // Copy assignment operator
    Frame& operator=( const Frame& rhs ) = delete;  // non copyable
    // move assignment operator
    Frame& operator=( Frame&& rhs ) = delete;  // movable
    // D'tor
    ~Frame() = default;

    /// @brief Select this frame as keyframe.
    void setKeyframe();

    /// @brief Add a feature to the image
    ///
    /// @param[in] feature New feature
    void addFeature( std::shared_ptr< Feature >& feature );

    /// @brief Remove a feature from this frame
    ///
    /// @param[in] feature Old feature that will be removed
    void removeFeature( std::shared_ptr< Feature >& feature );

    /// @brief Number of features
    ///
    /// @return std::size_t Size of features list
    std::size_t numberObservation() const;

    /// @brief If the point is in the front of camera (z>0) and can projectable into image
    ///  For this one, First the point transfers to camera cordinare with abs pose and then project into image
    ///  point3D_c = abs_pose * point3d_w
    ///  pixelPosition = K * point3D_c
    ///  check pixelPosition.x() and pixelPosition.y() with image size
    ///
    /// @param[in] point3d_w Position of a 3D point in world coordinate
    /// @return true
    /// @return false
    bool isVisible( const Eigen::Vector3d& point3d_w ) const;

    /// @brief Is this frame a keyframe
    ///
    /// @return true
    /// @return false
    bool isKeyframe() const;

    /// @brief Transform a 3D point from world to image pixel coordinate
    ///  point3d_c = abs_pose * point3d_w
    ///  pixelPosition = K * point3d_c
    ///
    /// @param[in] point3d_w Position of a 3D point in world coordinate
    /// @return Eigen::Vector2d 2D position in image pixel coordinate
    Eigen::Vector2d world2image( const Eigen::Vector3d& point3d_w ) const;

    /// @brief Transform a 3D point from world to camera coordinate
    ///  point3d_c = abs_pose * point3d_w
    ///
    /// @param[in] point3d_w Position of a 3D point in world coordinate
    /// @return Eigen::Vector3d 3D position in camera coordinate
    Eigen::Vector3d world2camera( const Eigen::Vector3d& point3d_w ) const;

    /// @brief Transform a 3D point from camera coordinate to world
    ///  point3d_w = abs_pose.inverse() * point3d_w
    ///
    /// @param[in] point3d_c Position of a 3D point in camera coordinate
    /// @return Eigen::Vector3d 3D position in world coordinate
    Eigen::Vector3d camera2world( const Eigen::Vector3d& point3d_c ) const;

    /// @brief Project a 3D point from camera coordinate to image pixel coordinate
    ///  pixelPosition = K * point3d_c
    ///
    /// @param[in] point3d_c Position of a 3D point in camera coordinate
    /// @return Eigen::Vector2d 2D position in image pixel coordinate
    Eigen::Vector2d camera2image( const Eigen::Vector3d& point3d_c ) const;

    /// @brief Inverse project a 2D point from image pixel coordinate to world cooridnate
    ///  point3d_c = K.inverse() * pixelPosition * depth
    ///  point3d_w = abs_pose.inverse() * point3d_c
    ///
    /// @param[in] pixelPosition Position of a 2D point in image pixel coordinate
    /// @param[in] depth depth of 2D point
    /// @return Eigen::Vector3d 3D position in world coordinate
    Eigen::Vector3d image2world( const Eigen::Vector2d& pixelPosition, const double depth ) const;

    /// @brief Inverse project a 2D point from image pixel cooridnate to camera coordinate
    ///  point3d_c = K.inverse() * pixelPosition * depth

    /// @param[in] pixelPosition Position of a 2D point in image pixel coordinate
    /// @param[in] depth depth of 2D point
    /// @return Eigen::Vector3d 3D position in camera coordinate
    Eigen::Vector3d image2camera( const Eigen::Vector2d& pixelPosition, const double depth ) const;

    /// @brief 3D position of camera in world cordinate
    ///  C = -R.transpose() * t
    ///  R.transpose() = R.inverse()
    ///
    /// @return Eigen::Vector3d 3D position of camera in world coordinate
    Eigen::Vector3d cameraInWorld() const;

    // TODO: reset frame

    static uint64_t m_frameCounter;                   ///< Counts the number of created frames. Used to set the unique id
    uint64_t m_id;                                    ///< Unique id of the frame
    const std::shared_ptr< PinholeCamera > m_camera;  ///< Camera model
    Sophus::SE3d m_absPose;                      ///< Estimated 3D rigid body pose ([R|t]) to transforms from world to camera (W2C)
    Eigen::Matrix< double, 6, 6 > m_covPose;          ///< Covariance matrix of pose. http://people.duke.edu/~hpgavin/ce281/lm.pdf, eq. 21
    ImagePyramid m_imagePyramid;                      ///< Image pyramid of image
    std::vector< std::shared_ptr< Feature > > m_features;  ///< List of features in the image
    bool m_keyFrame;                                       ///< Is this frames selected as keyframe?
    uint64_t m_timestamp;                                    ///< Timestamp of when the image was recorded
    std::shared_ptr <Frame> m_lastKeyframe;
    g2o::VertexSE3Expmap* m_optG2oFrame;

private:
};

#endif /* FRAME_HPP */