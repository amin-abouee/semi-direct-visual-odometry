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

#include <any>
#include <iostream>
#include <string>
#include <unordered_map>

#include <Eigen/Core>
#include <opencv2/core.hpp>

#include "frame.hpp"
#include "matplotlibcpp.h"
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

namespace visualization
{
// https://cloford.com/resources/colours/500col.htm
// https://www.w3schools.com/colors/colors_w3css.asp
const std::unordered_map< std::string, cv::Scalar > colors{
  {"red", cv::Scalar( 65, 82, 226 )},         {"pink", cv::Scalar( 101, 57, 215 )},   {"purple", cv::Scalar( 170, 55, 144 )},
  {"deep-purple", cv::Scalar( 177, 65, 96 )}, {"indigo", cv::Scalar( 175, 84, 65 )},  {"blue", cv::Scalar( 236, 150, 70 )},
  {"cyan", cv::Scalar( 209, 186, 83 )},       {"aqua", cv::Scalar( 253, 252, 115 )},  {"teal", cv::Scalar( 136, 148, 65 )},
  {"green", cv::Scalar( 92, 172, 103 )},      {"lime", cv::Scalar( 89, 218, 209 )},   {"yellow", cv::Scalar( 96, 234, 253 )},
  {"amber", cv::Scalar( 68, 194, 246 )},      {"orange", cv::Scalar( 56, 156, 242 )}, {"brown", cv::Scalar( 74, 86, 116 )},
  {"gray", cv::Scalar( 158, 158, 158 )},      {"black", cv::Scalar( 0, 0, 0 )},       {"deep-orange", cv::Scalar( 55, 99, 237 )},
  {"white", cv::Scalar( 256, 256, 256 )}};

namespace plt = matplotlibcpp;

auto drawingCircle = []( cv::Mat& img, const Eigen::Vector2d& point, const u_int32_t size, const cv::Scalar& color ) -> void {
    cv::circle( img, cv::Point2d( point.x(), point.y() ), size, color );
};

auto drawingRectangle = []( cv::Mat& img, const Eigen::Vector2d& point, const u_int32_t size, const cv::Scalar& color ) -> void {
    const u_int32_t halfPatch = size / 2;
    cv::rectangle( img, cv::Point2d( point.x() - halfPatch, point.y() - halfPatch ),
                   cv::Point2d( point.x() + halfPatch, point.y() + halfPatch ), color );
};

auto drawingLine = []( cv::Mat& img, const Eigen::Vector2d& point1, const Eigen::Vector2d& point2, const cv::Scalar& color ) -> void {
    cv::line( img, cv::Point2d( point1.x(), point1.y() ), cv::Point2d( point2.x(), point2.y() ), color );
};

void templatePatches( const cv::Mat& patches,
                      const uint32_t numberPatches,
                      const uint32_t patchSize,
                      const uint32_t horizontalMargin,
                      const uint32_t verticalMargin,
                      const uint32_t maxPatchInRow );

cv::Mat residualsPatches( const Eigen::VectorXd& residuals,
                          const uint32_t numberPatches,
                          const uint32_t patchSize,
                          const uint32_t horizontalMargin,
                          const uint32_t verticalMargin,
                          const uint32_t maxPatchInRow );

void grayImage( const cv::Mat& img, const std::string& windowsName );

void HSVColoredImage( const cv::Mat& img, const std::string& windowsName );

cv::Scalar generateColor( const float min, const float max, const float value );

cv::Mat getBGRImage( const cv::Mat& img );

void drawHistogram( std::map< std::string, std::any >& pack );

void featurePoints(
  cv::Mat& img,
  const Frame& frame,
  const u_int32_t radiusSize,
  const std::string& color,
  const std::function< void( cv::Mat& img, const Eigen::Vector2d& point, const u_int32_t size, const cv::Scalar& color ) >&
    drawingFunctor );

/// visualize feature points in any input image (for instance on HSV image)
void imageGrid( cv::Mat& img, const Frame& frame, const int32_t gridSize, const std::string& color );

void project3DPoints(
  cv::Mat& img,
  const Frame& frame,
  const u_int32_t radiusSize,
  const std::string& color,
  const std::function< void( cv::Mat& img, const Eigen::Vector2d& point, const u_int32_t size, const cv::Scalar& color ) >&
    drawingFunctor );

void projectPointsWithRelativePose(
  cv::Mat& img,
  const Frame& refFrame,
  const Frame& curFrame,
  const u_int32_t radiusSize,
  const std::string& color,
  const std::function< void( cv::Mat& img, const Eigen::Vector2d& point, const u_int32_t size, const cv::Scalar& color ) >&
    drawingFunctor );

void projectLinesWithRelativePose(
  cv::Mat& img,
  const Frame& refFrame,
  const Frame& curFrame,
  const uint32_t rangeInPixels,
  const std::string& color,
  const std::function< void( cv::Mat& img, const Eigen::Vector2d& point1, const Eigen::Vector2d& point2, const cv::Scalar& color ) >&
    drawingFunctor );

void projectLinesWithF(
  cv::Mat& img,
  const Frame& refFrame,
  const Eigen::Matrix3d& F,
  const uint32_t rangeInPixels,
  const std::string& color,
  const std::function< void( cv::Mat& img, const Eigen::Vector2d& point1, const Eigen::Vector2d& point2, const cv::Scalar& color ) >&
    drawingFunctor );

void epipole( cv::Mat& img,
              const Frame& frame,
              const u_int32_t radiusSize,
              const std::string& color,
              const std::function< void( cv::Mat& img, const Eigen::Vector2d& point, const u_int32_t size, const cv::Scalar& color ) >&
                drawingFunctor );

void stickTwoImageVertically( const cv::Mat& refImg, const cv::Mat& curImg, cv::Mat& img );

void stickTwoImageHorizontally( const cv::Mat& refImg, const cv::Mat& curImg, cv::Mat& img );

}  // namespace visualization

#endif /* __VISUALIZATION_H__ */