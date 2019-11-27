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

// class Visualization final
// {
// public:
//     // C'tor
//     explicit Visualization() = default;
//     // Copy C'tor
//     Visualization( const Visualization& rhs ) = default;
//     // move C'tor
//     Visualization( Visualization&& rhs ) = default;
//     // Copy assignment operator
//     Visualization& operator=( const Visualization& rhs ) = default;
//     // move assignment operator
//     Visualization& operator=( Visualization&& rhs ) = default;
//     // D'tor
//     ~Visualization() = default;

namespace Visualization
{
    void featurePoints( const Frame& frame, const std::string& windowsName );

    void featurePointsInBothImages( const Frame& refFrame, const Frame& curFrame, const std::string& windowsName );

    void featurePointsInBothImagesWithSearchRegion( const Frame& refFrame,
                                                    const Frame& curFrame,
                                                    const uint16_t& patchSize,
                                                    const std::string& windowsName );

    void featurePoints( const cv::Mat& img, const Frame& frame, const std::string& windowsName );

    void grayImage( const cv::Mat& img, const std::string& windowsName );

    void HSVColoredImage( const cv::Mat& img, const std::string& windowsName );

    void epipole( const Frame& frame, const Eigen::Vector3d& vec, const std::string& windowsName );

    void epipolarLine( const Frame& frame,
                    const Eigen::Vector3d& vec,
                    const Eigen::Matrix3d& F,
                    const std::string& windowsName );

    void epipolarLine( const Frame& curFrame,
                    const Eigen::Vector3d& normalizedVec,
                    const double minDepth,
                    const double maxDepth,
                    const std::string& windowsName );

    void epipolarLine( const Frame& refFrame,
                    const Frame& curFrame,
                    const Eigen::Vector2d& feature,
                    const double minDepth,
                    const double maxDepth,
                    const std::string& windowsName );

    void epipolarLinesWithFundamenalMatrix( const Frame& frame,
                                            const cv::Mat& currentImg,
                                            const Eigen::Matrix3d& F,
                                            const std::string& windowsName );

    void epipolarLinesWithEssentialMatrix( const Frame& frame,
                                        const cv::Mat& currentImg,
                                        const Eigen::Matrix3d& E,
                                        const std::string& windowsName );

    // private:
    cv::Scalar generateColor( const double min, const double max, const float value );
    cv::Mat getBGRImage( const cv::Mat& img );
};  // namespace Visualization

#endif /* __VISUALIZATION_H__ */