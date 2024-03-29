#ifndef __PINHOLE_CAMERA_HPP__
#define __PINHOLE_CAMERA_HPP__

#include <iostream>

#include <Eigen/Core>
#include <opencv2/core.hpp>


// https://github.com/uzh-rpg/rpg_vikit/blob/10871da6d84c8324212053c40f468c6ab4862ee0/vikit_common/include/vikit/abstract_camera.h
// https://github.com/uzh-rpg/rpg_vikit/blob/10871da6d84c8324212053c40f468c6ab4862ee0/vikit_common/include/vikit/pinhole_camera.h
/**
 * @brief This class conatains the pinhole camera information: the camera calibration matrix, lens distortion, and inverse projection
 * 
 */
class PinholeCamera final
{
public:
EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    explicit PinholeCamera( const int32_t width,
                            const int32_t height,
                            const double fx,
                            const double fy,
                            const double cx,
                            const double cy,
                            const double d0,
                            const double d1,
                            const double d2,
                            const double d3,
                            const double d4 );

    explicit PinholeCamera (const int32_t width,
                            const int32_t height,
                            const cv::Mat& cameraMatrix,
                            const cv::Mat& distortionCoeffs );

    PinholeCamera(const PinholeCamera& rhs) = delete;
    PinholeCamera(PinholeCamera&& rhs) = delete;
    PinholeCamera& operator=(const PinholeCamera& rhs) = delete;
    PinholeCamera& operator=(PinholeCamera&& rhs) = delete;
    ~PinholeCamera() = default;

    /// project 3D points in camera coordinate to image coordinate
    Eigen::Vector2d project2d(const double x, const double y, const double z) const;
    Eigen::Vector2d project2d(const Eigen::Vector3d& pointImage) const;

    /// inverse project a point in image to camera coordinate
    Eigen::Vector3d inverseProject2d(const Eigen::Vector2d& pointInCamera) const;
    Eigen::Vector3d inverseProject2d(const double x, const double y) const;

    const Eigen::Matrix3d& K() const;
    const Eigen::Matrix3d& invK() const;
    const cv::Mat& K_cv() const;

    double fx() const;
    double fy() const;
    double cx() const;
    double cy() const;

    Eigen::Vector2d focalLength() const;
    Eigen::Vector2d principalPoint() const;
    
    int32_t width() const;
    int32_t height() const;

    bool isInFrame(const Eigen::Vector2d& imagePoint, const double boundary = 0.0 ) const;
    bool isInFrame(const Eigen::Vector2d& imagePoint, const uint8_t level, const double boundary ) const;

    void undistortImage(const cv::Mat& distorted, cv::Mat& undistorted);

private:
    int32_t m_width;
    int32_t m_height;
    Eigen::Matrix3d m_K;
    Eigen::Matrix3d m_invK;
    Eigen::Matrix<double, 1, 5> m_distortion;
    cv::Mat m_cvK;
    cv::Mat m_cvDistortion;
    cv::Mat undistortedMapX;
    cv::Mat undistortedMapY;
    bool m_applyDistortion;
};

#endif /* __PINHOLE_CAMERA_HPP__ */