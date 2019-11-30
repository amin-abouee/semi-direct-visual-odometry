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
#include <map>
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
// https://cloford.com/resources/colours/500col.htm
// https://www.w3schools.com/colors/colors_w3css.asp
const std::map< std::string, cv::Scalar > colors{
  {"red", cv::Scalar( 65, 82, 226 )},     {"pink", cv::Scalar( 101, 57, 215 )},
  {"purple", cv::Scalar( 170, 55, 144 )}, {"deep-purple", cv::Scalar( 177, 65, 96 )},
  {"indigo", cv::Scalar( 175, 84, 65 )},  {"blue", cv::Scalar( 236, 150, 70 )},
  {"cyan", cv::Scalar( 209, 186, 83 )},   {"aqua", cv::Scalar( 253, 252, 115 )},
  {"teal", cv::Scalar( 136, 148, 65 )},   {"green", cv::Scalar( 92, 172, 103 )},
  {"lime", cv::Scalar( 89, 218, 209 )},   {"yellow", cv::Scalar( 96, 234, 253 )},
  {"amber", cv::Scalar( 68, 194, 246 )},  {"orange", cv::Scalar( 56, 156, 242 )},
  {"brown", cv::Scalar( 74, 86, 116 )},   {"gray", cv::Scalar( 158, 158, 158 )},
  {"black", cv::Scalar( 0, 0, 0 )},       {"deep-orange", cv::Scalar( 55, 99, 237 )},
  {"white", cv::Scalar( 356, 256, 256 )}};

/// visualize feature points in frame image
void featurePoints( const Frame& frame, const std::string& windowsName );

/// visualize feature points in any input image (for instance on HSV image)
void featurePoints( const cv::Mat& img, const Frame& frame, const std::string& windowsName );

/// visualize feature points in both frames. images stick to each other
void featurePointsInBothImages( const Frame& refFrame, const Frame& curFrame, const std::string& windowsName );

/// visualize feature points with a search area around that in current frame and corresponding points in reference image
void featurePointsInBothImagesWithSearchRegion( const Frame& refFrame,
                                                const Frame& curFrame,
                                                const uint16_t& patchSize,
                                                const std::string& windowsName );

/// draw epipole point in input image
void epipole( const Frame& frame, const Eigen::Vector3d& vec, const std::string& windowsName );

/// draw epipolar line for a specific bearing vec in the range of image
void epipolarLine( const Frame& frame,
                   const Eigen::Vector3d& vec,
                   const Eigen::Matrix3d& F,
                   const std::string& windowsName );

/// draw epipolar line for a specific bearing vector in a specific range of depth
void epipolarLine( const Frame& curFrame,
                   const Eigen::Vector3d& normalizedVec,
                   const double minDepth,
                   const double maxDepth,
                   const std::string& windowsName );

/// draw epipolar line for a specific feature point in a specific range of depth
void epipolarLine( const Frame& refFrame,
                   const Frame& curFrame,
                   const Eigen::Vector2d& feature,
                   const double minDepth,
                   const double maxDepth,
                   const std::string& windowsName );

/// draw all epipolar lines with fundamental matrix
void epipolarLinesWithFundamentalMatrix( const Frame& frame,
                                         const cv::Mat& currentImg,
                                         const Eigen::Matrix3d& F,
                                         const std::string& windowsName );

void epipolarLinesWithPointsWithFundamentalMatrix( const Frame& refFrame,
                                                   const Frame& curframe,
                                                   const Eigen::Matrix3d& F,
                                                   const std::string& windowsName );

/// draw all epipolar lines with essential matrix
void epipolarLinesWithEssentialMatrix( const Frame& frame,
                                       const cv::Mat& currentImg,
                                       const Eigen::Matrix3d& E,
                                       const std::string& windowsName );

void grayImage( const cv::Mat& img, const std::string& windowsName );

void HSVColoredImage( const cv::Mat& img, const std::string& windowsName );

cv::Scalar generateColor( const double min, const double max, const float value );

cv::Mat getBGRImage( const cv::Mat& img );

};  // namespace Visualization

#endif /* __VISUALIZATION_H__ */