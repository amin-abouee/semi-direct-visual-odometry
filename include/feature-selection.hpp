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

#include <opencv2/core.hpp>

#include <Eigen/Core>

class FeatureSelection final
{
public:
    // C'tor
    explicit FeatureSelection( const cv::Mat& imgGray );

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

    void detectFeatures( const uint32_t numberCandidate );

    // void visualizeFeaturePoints();

    // void visualizeGrayGradientMagnitude();

    // void visualizeColoredGradientMagnitude();

    // void visualizeEpipolar( const Eigen::Vector3d& point, const Eigen::Matrix3d& K );

    Eigen::Matrix< double, 3, Eigen::Dynamic > m_kp;
    cv::Mat m_gradientMagnitude;
    cv::Mat m_gradientOrientation;

private:
    // Efficient adaptive non-maximal suppression algorithms for homogeneous spatial keypoint distribution
    Eigen::Matrix< double, 3, Eigen::Dynamic > Ssc(
      std::vector< cv::KeyPoint > keyPoints, int numRetPoints, float tolerance, int cols, int rows );

    // cv::Scalar generateColor( const double min, const double max, const float value );

    cv::Mat m_dx;
    cv::Mat m_dy;

    // uint32_t m_numberCandidate;

    std::shared_ptr< cv::Mat > m_imgGray;
    // std::vector < cv::KeyPoint > m_keypoints;

};

#endif /* __FEATURE_SELECTION_H__ */