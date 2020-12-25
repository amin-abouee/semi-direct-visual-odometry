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
#include <memory>
#include <string>
#include <unordered_map>

#include <Eigen/Core>
#include <opencv2/core.hpp>

#include "depth_estimator.hpp"
#include "frame.hpp"
#include "matplotlibcpp.h"

namespace visualization
{
// https://cloford.com/resources/colours/500col.htm
// https://www.w3schools.com/colors/colors_w3css.asp
const std::unordered_map< std::string, cv::Scalar > colors{
  { "red", cv::Scalar( 65, 82, 226 ) },         { "pink", cv::Scalar( 101, 57, 215 ) },   { "purple", cv::Scalar( 170, 55, 144 ) },
  { "deep-purple", cv::Scalar( 177, 65, 96 ) }, { "indigo", cv::Scalar( 175, 84, 65 ) },  { "blue", cv::Scalar( 236, 150, 70 ) },
  { "cyan", cv::Scalar( 209, 186, 83 ) },       { "aqua", cv::Scalar( 253, 252, 115 ) },  { "teal", cv::Scalar( 136, 148, 65 ) },
  { "green", cv::Scalar( 92, 172, 103 ) },      { "lime", cv::Scalar( 89, 218, 209 ) },   { "yellow", cv::Scalar( 96, 234, 253 ) },
  { "amber", cv::Scalar( 68, 194, 246 ) },      { "orange", cv::Scalar( 56, 156, 242 ) }, { "brown", cv::Scalar( 74, 86, 116 ) },
  { "gray", cv::Scalar( 158, 158, 158 ) },      { "black", cv::Scalar( 0, 0, 0 ) },       { "deep-orange", cv::Scalar( 55, 99, 237 ) },
  { "white", cv::Scalar( 256, 256, 256 ) } };

namespace plt = matplotlibcpp;

auto inline drawingCircle = []( cv::Mat& img, const Eigen::Vector2d& point, const u_int32_t size, const cv::Scalar& color ) -> void {
    cv::circle( img, cv::Point2d( point.x(), point.y() ), size, color );
};

auto inline drawingRectangle = []( cv::Mat& img, const Eigen::Vector2d& point, const u_int32_t size, const cv::Scalar& color ) -> void {
    cv::rectangle( img, cv::Point2d( point.x() - size, point.y() - size ), cv::Point2d( point.x() + size, point.y() + size ), color );
};

auto inline drawingLine = [](
                            cv::Mat& img, const Eigen::Vector2d& point1, const Eigen::Vector2d& point2, const cv::Scalar& color ) -> void {
    cv::line( img, cv::Point2d( point1.x(), point1.y() ), cv::Point2d( point2.x(), point2.y() ), color );
};

cv::Mat getGrayImage( const cv::Mat& img );

cv::Mat getColorImage( const cv::Mat& img );

cv::Mat getHSVImageWithMagnitude( const cv::Mat& img, const uint8_t minMagnitude );

cv::Scalar generateColor( const uint8_t min, const uint8_t max, const uint8_t value );

void stickTwoImageVertically( const cv::Mat& refImg, const cv::Mat& curImg, cv::Mat& img );

void stickTwoImageHorizontally( const cv::Mat& refImg, const cv::Mat& curImg, cv::Mat& img );

void featurePoints(
  cv::Mat& img,
  const std::shared_ptr< Frame >& frame,
  const u_int32_t radiusSize,
  const std::string& color,
  const bool justFeatureWithoutVisiblePoint,
  const std::function< void( cv::Mat& img, const Eigen::Vector2d& point, const u_int32_t size, const cv::Scalar& color ) >&
    drawingFunctor );

/// visualize feature points in any input image (for instance on HSV image)
void imageGrid( cv::Mat& img, const int32_t gridSize, const std::string& color );

///@brief
///
///@param img
///@param frame
///@param radiusSize
///@param color
void colormapDepth( cv::Mat& img, const std::shared_ptr< Frame >& frame, const u_int32_t radiusSize, const std::string& color );

void projectPointsWithRelativePose(
  cv::Mat& img,
  const std::shared_ptr< Frame >& refFrame,
  const std::shared_ptr< Frame >& curFrame,
  const u_int32_t radiusSize,
  const std::string& color,
  const std::function< void( cv::Mat& img, const Eigen::Vector2d& point, const u_int32_t size, const cv::Scalar& color ) >&
    drawingFunctor );

void projectLinesWithRelativePose(
  cv::Mat& img,
  const std::shared_ptr< Frame >& refFrame,
  const std::shared_ptr< Frame >& curFrame,
  const uint32_t rangeInPixels,
  const std::string& color,
  const std::function< void( cv::Mat& img, const Eigen::Vector2d& point1, const Eigen::Vector2d& point2, const cv::Scalar& color ) >&
    drawingFunctor );

void projectLinesWithF(
  cv::Mat& img,
  const std::shared_ptr< Frame >& refFrame,
  const Eigen::Matrix3d& F,
  const uint32_t rangeInPixels,
  const std::string& color,
  const std::function< void( cv::Mat& img, const Eigen::Vector2d& point1, const Eigen::Vector2d& point2, const cv::Scalar& color ) >&
    drawingFunctor );

void epipole( cv::Mat& img,
              const std::shared_ptr< Frame >& frame,
              const u_int32_t radiusSize,
              const std::string& color,
              const std::function< void( cv::Mat& img, const Eigen::Vector2d& point, const u_int32_t size, const cv::Scalar& color ) >&
                drawingFunctor );

cv::Mat referencePatches( const cv::Mat& patches,
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

void drawHistogram( std::map< std::string, std::any >& pack );

void projectDepthFilters(
  cv::Mat& img,
  const std::shared_ptr< Frame >& frame,
  const std::vector< MixedGaussianFilter >& depthFilters,
  const u_int32_t radiusSize,
  const std::string& color,
  const std::function< void( cv::Mat& img, const Eigen::Vector2d& point1, const Eigen::Vector2d& point2, const cv::Scalar& color ) >&
    drawingFunctor );

void projectDepthFilters(
  cv::Mat& img,
  const std::shared_ptr< Frame >& frame,
  const std::vector< MixedGaussianFilter >& depthFilters,
  const std::vector< double >& updatedDepths,
  const u_int32_t radiusSize,
  const std::string& color,
  const std::function< void( cv::Mat& img, const Eigen::Vector2d& point1, const Eigen::Vector2d& point2, const cv::Scalar& color ) >&
    drawingFunctor );

}  // namespace visualization

#endif /* __VISUALIZATION_H__ */