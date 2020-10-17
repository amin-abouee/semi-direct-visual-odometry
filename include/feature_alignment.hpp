#ifndef __FEATURE_ALIGNMENT_HPP__
#define __FEATURE_ALIGNMENT_HPP__

#include <iostream>
#include <memory>
#include <vector>

#include <Eigen/Core>
#include <opencv2/core.hpp>
#include <sophus/se2.hpp>

#include "frame.hpp"
#include "optimizer.hpp"

class FeatureAlignment
{
public:
    explicit FeatureAlignment( uint32_t patchSize, int32_t level, uint32_t numParameters );
    FeatureAlignment( const FeatureAlignment& rhs );
    FeatureAlignment( FeatureAlignment&& rhs );
    FeatureAlignment& operator=( const FeatureAlignment& rhs );
    FeatureAlignment& operator=( FeatureAlignment&& rhs );
    ~FeatureAlignment()       = default;

    double align( const std::shared_ptr< Feature >& refFeature, const std::shared_ptr< Frame >& curFrame, Eigen::Vector2d& pixelPos );

private:
    uint32_t m_patchSize;
    uint32_t m_halfPatchSize;
    uint32_t m_patchArea;
    uint32_t m_currentLevel;
    int32_t m_level;

    Optimizer m_optimizer;
    cv::Mat m_refPatches;
    std::vector< bool > m_refVisibility;

    void computeJacobian( const std::shared_ptr< Feature >& refFeature);
    uint32_t computeResiduals( const std::shared_ptr< Feature >& refFeature,
                               const std::shared_ptr< Frame >& curFrame,
                               Sophus::SE2d& pose );
    void computeImageJac( Eigen::Matrix< double, 2, 3 >& imageJac, const Eigen::Vector2d& pixelPos );
    void update( Sophus::SE2d& pose, const Eigen::Vector3d& dx );
    void resetParameters();
};

#endif /* __FEATURE_ALIGNMENT_HPP__ */