#ifndef __BUNDLE_ADJUSTMENT_HPP__
#define __BUNDLE_ADJUSTMENT_HPP__

#include <iostream>
#include <memory>
#include <vector>

#include <Eigen/Core>
#include <opencv2/core.hpp>
#include <sophus/se3.hpp>

#include "frame.hpp"
#include "point.hpp"
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
    double optimizeStructure( std::shared_ptr< Frame >& frame, const uint32_t maxNumberPoints );

private:
    uint32_t m_currentLevel;
    int32_t m_level;

    Optimizer m_optimizer;
    std::vector< bool > m_refVisibility;

    void computeJacobianPose( const std::shared_ptr< Frame >& frame );
    void computeImageJacPose( Eigen::Matrix< double, 2, 6 >& imageJac, const Eigen::Vector3d& point, const double fx, const double fy );
    uint32_t computeResidualsPose( const std::shared_ptr< Frame >& frame, const Sophus::SE3d& pose );
    void updatePose( Sophus::SE3d& pose, const Eigen::VectorXd& dx );

    void computeJacobianStructure( const std::shared_ptr< Point >& point );
    void computeImageJacStructure( Eigen::Matrix< double, 2, 3 >& imageJac,
                                   const Eigen::Vector3d& point,
                                   const Eigen::Matrix3d& rotation,
                                   const double fx,
                                   const double fy );
    uint32_t computeResidualsStructure( const std::shared_ptr< Point >& point );
    void updateStructure( const std::shared_ptr< Point >& point, const Eigen::Vector3d& dx );

    void resetParameters();
};

#endif /* __BUNDLE_ADJUSTMENT_HPP__ */