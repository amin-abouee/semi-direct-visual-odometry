/**
 * @file mixed_gaussian_filter.hpp
 * @brief mixed representation of depth based on gaussian adn beta distribution
 *
 * @date 11.05.2020
 * @author Amin Abouee
 *
 * @section DESCRIPTION
 *
 *
 */
#ifndef __MIXED_GAUSSIAN_FILTER_H__
#define __MIXED_GAUSSIAN_FILTER_H__

#include <iostream>
#include <memory>

#include "feature.hpp"

class MixedGaussianFilter final
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    static uint64_t m_frameCounter;
    static uint64_t m_filterCounter;

    uint64_t m_frameId;                    //!< Frame id is the id of the keyframe for which the filter was created.
    uint64_t m_id;                         //!< Filter ID, only used for visualization.
    std::shared_ptr< Feature > m_feature;  //!< Feature in the keyframe for which the depth should be computed.
    double m_a;                            //!< a of Beta distribution: When high, probability of inlier is large.
    double m_b;                            //!< b of Beta distribution: When high, probability of outlier is large.
    double m_mu;                           //!< Mean of normal distribution.
    double m_maxDepth;                     //!< Max range of the possible depth.
    double m_var;                          //!< Variance of normal distribution.
    double m_sigma;                        //!< sigma of normal distribution.
    Eigen::Matrix2d m_CovPatch;            //!< Patch covariance in reference image.
    bool m_validity;                       //!< Check the validity of this filter

    // C'tor
    explicit MixedGaussianFilter(const std::shared_ptr< Feature >& feature, const double depthMean, const double depthMin);
    // Copy C'tor
    MixedGaussianFilter( const MixedGaussianFilter& rhs ) = default;
    // move C'tor
    MixedGaussianFilter( MixedGaussianFilter&& rhs ) = default;
    // Copy assignment operator
    MixedGaussianFilter& operator=( const MixedGaussianFilter& rhs ) = default;
    // move assignment operator
    MixedGaussianFilter& operator=( MixedGaussianFilter&& rhs ) = default;
    // D'tor
    ~MixedGaussianFilter() = default;

private:
};

#endif /* __MIXED_GAUSSIAN_FILTER_H__ */