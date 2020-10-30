/**
 * @file feature-selection.hpp
 * @brief how to select features in image
 *
 * @date 02.11.2019
 * @author Amin Abouee
 *
 * @section DESCRIPTION
 *
 *
 */
#ifndef __FEATURE_SELECTION_H__
#define __FEATURE_SELECTION_H__

#include <iostream>
#include <memory>
#include <vector>

#include <Eigen/Core>
#include <opencv2/core.hpp>

#include "feature.hpp"

class FeatureSelection final
{
public:
    // C'tor
    // explicit FeatureSelection( const cv::Mat& imgGray );
    explicit FeatureSelection( const int32_t width, const int32_t height, const uint32_t cellSize );

    // explicit FeatureSelection( const cv::Mat& imgGray );

    // explicit FeatureSelection( const cv::Mat& imgGray, const uint32_t numberCandidate );

    // Copy C'tor
    FeatureSelection( const FeatureSelection& rhs ) = default;

    // move C'tor
    FeatureSelection( FeatureSelection&& rhs ) = default;

    // Copy assignment operator
    FeatureSelection& operator=( const FeatureSelection& rhs ) = default;

    // move assignment operator
    FeatureSelection& operator=( FeatureSelection&& rhs ) = default;

    // D'tor
    ~FeatureSelection() = default;

    void detectFeaturesWithSSC( std::shared_ptr< Frame >& frame, const uint32_t numberCandidate );

    void detectFeaturesInGrid( std::shared_ptr< Frame >& frame, const float detectionThreshold );

    void detectFeaturesByValue( std::shared_ptr< Frame >& frame, const float detectionThreshold );

    void setExistingFeatures (const std::vector<std::shared_ptr<Feature>>& features);

    void setCellInGridOccupancy(const Eigen::Vector2d& location);

    // std::vector< Feature > m_features;
    cv::Mat m_gradientMagnitude;
    cv::Mat m_gradientOrientation;

    uint32_t m_cellSize;
    uint32_t m_gridCols;
    uint32_t m_gridRows;
    std::vector< bool > m_occupancyGrid;

private:
    // Efficient adaptive non-maximal suppression algorithms for homogeneous spatial keypoint distribution
    void Ssc( std::shared_ptr< Frame >& frame,
              const std::vector< cv::KeyPoint >& keyPoints,
              const uint32_t numRetPoints,
              const float tolerance,
              const uint32_t cols,
              const uint32_t rows );

    void comouteImageGradient( const cv::Mat& imgGray, cv::Mat& imgGradientMagnitude, cv::Mat& imgGradientOrientation );

    void resetGridOccupancy ();

    // void computeGradient( const cv::Mat& currentTemplateImage, cv::Mat& templateGradientX, cv::Mat& templateGradientY );
};

#endif /* __FEATURE_SELECTION_H__ */