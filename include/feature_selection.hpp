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
#ifndef FEATURE_SELECTION_HPP
#define FEATURE_SELECTION_HPP

#include "feature.hpp"

#include <Eigen/Core>
#include <opencv2/core.hpp>

#include <iostream>
#include <memory>
#include <vector>

///@brief Feature selection class, contains the methods to extract features from the image gradient
class FeatureSelection final
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    ///@brief Construct a new Feature Selection object
    ///
    ///@param[in] width
    ///@param[in] height
    ///@param[in] cellSize
    explicit FeatureSelection( const int32_t width, const int32_t height, const int32_t cellSize );

    // copy C'tor
    FeatureSelection( const FeatureSelection& rhs ) = delete;

    // move C'tor
    FeatureSelection( FeatureSelection&& rhs ) = delete;

    // Copy assignment operator
    FeatureSelection& operator=( const FeatureSelection& rhs ) = delete;

    // move assignment operator
    FeatureSelection& operator=( FeatureSelection&& rhs ) = delete;

    // D'tor
    ~FeatureSelection() = default;

    ///@brief Select all pixels with gradient magnitude > threshold and then run Sup-pression via Square Covering
    /// to have normal distribution of points
    ///
    ///@param[in, out] frame Frame object
    ///@param[in] detectionThreshold Threshold for gradient magnitude
    ///@param[in] numberCandidate Maximum number of candidate
    ///@param[in] useBucketing enable/disable bucketing algorithm (grid)
    void gradientMagnitudeWithSSC( std::shared_ptr< Frame >& frame,
                                   const float detectionThreshold,
                                   const uint32_t numberCandidate,
                                   const bool useBucketing = true );

    ///@brief Compute image gradient of inout frame and select all points that have gradient magnitude > threshold.
    ///
    ///@param[in, out] frame Frame object
    ///@param[in] detectionThreshold Threshold for gradient magnitude
    ///@param[in] useBucketing enable/disable bucketing algorithm (grid)
    void gradientMagnitudeByValue( std::shared_ptr< Frame >& frame, const float detectionThreshold, const bool useBucketing = true );

    ///@brief Set the Existing feature in corresponding cell
    ///
    ///@param[in] features Feature object
    void setExistingFeatures( const std::vector< std::shared_ptr< Feature > >& features );

    ///@brief Set the Cell In Grid Occupancy object
    ///
    ///@param[in] pixelPosition Position of feature point
    void setCellInGridOccupancy( const Eigen::Vector2d& pixelPosition );

    cv::Mat m_imgGradientMagnitude;       ///< Gradient magnitute of input image (base image)
    cv::Mat m_imgGradientOrientation;     ///< Gradient orientation of input image (base image)
    int32_t m_cellSize;                   ///< Cell size in pixel, when we have a grid
    int32_t m_gridCols;                   ///< Number of grid columns
    int32_t m_gridRows;                   ///< Number of grid rows
    std::vector< bool > m_occupancyGrid;  ///< Grid of bool, when we have a feature in one cell, corresponding cell is 1, otherwise 0

private:
    ///@brief Efficient adaptive non-maximal suppression algorithms for homogeneous spatial keypoint distribution
    /// SSC: sup-pression via Square Covering
    ///@link https://github.com/BAILOOL/ANMS-Codes
    ///
    ///@param[in, out] frame Pointer to keyframe
    ///@param[in] keyPoints List of all candidate, sorted by response
    ///@param[in] numRetPoints Maximum number of detected points
    ///@param[in] tolerance Tolerance around the numRetPoints
    ///@param[in] cols Number of grid cols
    ///@param[in] rows Number of grid rows
    void SSC( const std::vector< cv::KeyPoint >& keyPoints,
              const int32_t numRetPoints,
              const float tolerance,
              const int32_t cols,
              const int32_t rows,
              std::vector< int32_t >& ResultVec );

    ///@brief Compute the gradient magnitude and orientation of input image
    ///
    ///@param[in] imgGray input base image
    void computeImageGradient( const cv::Mat& imgGray );

    ///@brief Reset all cells of grid
    void resetGridOccupancy();
};

#endif /* FEATURE_SELECTION_HPP */