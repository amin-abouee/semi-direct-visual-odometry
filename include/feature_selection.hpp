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
#include <vector>
#include <memory>

#include <Eigen/Core>
#include <opencv2/core.hpp>

#include <spdlog/spdlog.h>

#include "feature.hpp"

class FeatureSelection final
{
public:
    // C'tor
    explicit FeatureSelection(const cv::Mat& imgGray);

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

    void detectFeaturesWithSSC( Frame& frame, const uint32_t numberCandidate );

    void detectFeaturesInGrid( Frame& frame, const int32_t gridSize );

    // std::vector< Feature > m_features;
    cv::Mat m_gradientMagnitude;
    cv::Mat m_gradientOrientation;

private:
    // Efficient adaptive non-maximal suppression algorithms for homogeneous spatial keypoint distribution
    void Ssc( Frame& frame,
              const std::vector< cv::KeyPoint >& keyPoints,
              const uint32_t numRetPoints,
              const float tolerance,
              const uint32_t cols,
              const uint32_t rows );

    // void computeGradient( const cv::Mat& currentTemplateImage, cv::Mat& templateGradientX, cv::Mat& templateGradientY );

    cv::Mat m_dx;
    cv::Mat m_dy;

    std::shared_ptr<spdlog::logger> featureLogger;

    // std::shared_ptr< cv::Mat > m_imgGray;

    // uint32_t m_numberFeatures;
};

#endif /* __FEATURE_SELECTION_H__ */