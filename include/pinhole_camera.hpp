#ifndef __PINHOLE_CAMERA_HPP__
#define __PINHOLE_CAMERA_HPP__

#include <iostream>

#include <Eigen/Core>
#include <opencv2/core.hpp>


// https://github.com/uzh-rpg/rpg_vikit/blob/10871da6d84c8324212053c40f468c6ab4862ee0/vikit_common/include/vikit/abstract_camera.h
// https://github.com/uzh-rpg/rpg_vikit/blob/10871da6d84c8324212053c40f468c6ab4862ee0/vikit_common/include/vikit/pinhole_camera.h
class PinholeCamera final
{
public:
EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    explicit PinholeCamera( const double width,
                            const double height,
                            const double fx,
                            const double fy,
                            const double cx,
                            const double cy,
                            const double d0,
                            const double d1,
                            const double d2,
                            const double d3,
                            const double d4 );

    explicit PinholeCamera (const double width,
                            const double height,
                            const cv::Mat& cameraMatrix,
                            const cv::Mat& distortionCoeffs );

    PinholeCamera(const PinholeCamera& rhs) = default;
    PinholeCamera(PinholeCamera&& rhs) =default;
    PinholeCamera& operator=(const PinholeCamera& rhs) = default;
    PinholeCamera& operator=(PinholeCamera&& rhs) = default;
    ~PinholeCamera() = default;

    Eigen::Vector2d project2d(const double x, const double y, const double z) const;
    Eigen::Vector2d project2d(const Eigen::Vector3d& pointImage) const;

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
    Eigen::Vector2d principlePoint() const;
    double width() const;
    double height() const;

    bool isInFrame(const Eigen::Vector2d& imagePoint, const double boundary = 0.0 ) const;
    bool isInFrame(const Eigen::Vector2d& imagePoint, const uint8_t level, const double boundary ) const;

    void undistortImage(const cv::Mat& distorted, cv::Mat& undistorted);

private:
    double m_width;
    double m_height;
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