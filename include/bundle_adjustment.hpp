#ifndef __BUNDLE_ADJUSTMENT_HPP__
#define __BUNDLE_ADJUSTMENT_HPP__

#include <iostream>
#include <memory>
#include <vector>

#include <Eigen/Core>
#include <opencv2/core.hpp>
#include <sophus/se3.hpp>

#include "frame.hpp"
#include "optimizer.hpp"

class BundleAdjustment
{
public:
    explicit BundleAdjustment( int32_t level, uint32_t numParameters );
    BundleAdjustment( const BundleAdjustment& rhs );
    BundleAdjustment( BundleAdjustment&& rhs );
    BundleAdjustment& operator=( const BundleAdjustment& rhs );
    BundleAdjustment& operator=( BundleAdjustment&& rhs );
    ~BundleAdjustment()       = default;

    double optimizePose( std::shared_ptr< Frame >& frame );

private:
    uint32_t m_currentLevel;
    int32_t m_level;

    Optimizer m_optimizer;
    std::vector< bool > m_refVisibility;

    void computeJacobianPose( std::shared_ptr< Frame >& frame );
    uint32_t computeResidualsPose( std::shared_ptr< Frame >& frame, Sophus::SE3d& pose );
    void computeImageJac( Eigen::Matrix< double, 2, 6 >& imageJac, const Eigen::Vector3d& point, const double fx, const double fy );
    void updatePose( Sophus::SE3d& pose, const Eigen::VectorXd& dx );
    void resetParameters();
};

#endif /* __BUNDLE_ADJUSTMENT_HPP__ */