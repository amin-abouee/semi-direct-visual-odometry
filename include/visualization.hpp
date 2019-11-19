/**
 * @file visualization.hpp
 * @brief visualization class for each cases
 *
 * @date 06.11.2019
 * @author Amin Abouee
 *
 * @section DESCRIPTION
 *
 *
 */
#ifndef __VISUALIZATION_H__
#define __VISUALIZATION_H__

#include <iostream>
#include <string>

#include "frame.hpp"

#include <Eigen/Core>
#include <opencv2/core.hpp>

class Visualization final
{
public:
    // C'tor
    explicit Visualization() = default;
    // Copy C'tor
    Visualization( const Visualization& rhs ) = default;
    // move C'tor
    Visualization( Visualization&& rhs ) = default;
    // Copy assignment operator
    Visualization& operator=( const Visualization& rhs ) = default;
    // move assignment operator
    Visualization& operator=( Visualization&& rhs ) = default;
    // D'tor
    ~Visualization() = default;

    void visualizeFeaturePoints( const Frame& frame, const std::string& windowsName );

    void visualizeFeaturePointsGradientMagnitude( const cv::Mat& img,
                                                  const Frame& frame,
                                                  const std::string& windowsName );

    void visualizeGrayImage( const cv::Mat& img, const std::string& windowsName );

    void visualizeHSVColoredImage( const cv::Mat& img, const std::string& windowsName );

    void visualizeEpipole( const Frame& frame, const Eigen::Vector3d& vec, const std::string& windowsName );

    void visualizeEpipolarLine( const Frame& frame,
                                const Eigen::Vector3d& vec,
                                const Eigen::Matrix3d& F,
                                const std::string& windowsName );

    void visualizeEpipolarLines( const Frame& frame, const Eigen::Matrix3d& F, const std::string& windowsName );

private:
    cv::Scalar generateColor( const double min, const double max, const float value );
    cv::Mat getBGRImage( const cv::Mat& img );
};

#endif /* __VISUALIZATION_H__ */