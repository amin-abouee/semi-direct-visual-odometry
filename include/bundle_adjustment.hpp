#ifndef __BUNDLE_ADJUSTMENT_HPP__
#define __BUNDLE_ADJUSTMENT_HPP__

#include "frame.hpp"
#include "map.hpp"
#include "optimizer.hpp"
#include "point.hpp"

#include <Eigen/Core>
#include <opencv2/core.hpp>
#include <sophus/se3.hpp>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <g2o/core/sparse_optimizer.h>

#include <iostream>
#include <memory>
#include <vector>

// namespace g2o
// {
// class EdgeProjectXYZ2UV;
// class SparseOptimizer;
// class VertexSE3Expmap;
// class VertexSBAPointXYZ;
// }  // namespace g2o

class BundleAdjustment
{
public:
    using g2oEdgeSE3  = g2o::EdgeProjectXYZ2UV;
    using g2oFrameSE3 = g2o::VertexSE3Expmap;
    using g2oPoint    = g2o::VertexPointXYZ;

    struct EdgeContainerSE3
    {
        g2oEdgeSE3* edge;
        std::shared_ptr< Frame > frame;
        std::shared_ptr< Feature > feature;
        bool is_deleted;
        EdgeContainerSE3( g2oEdgeSE3* e, std::shared_ptr< Frame >& frame, std::shared_ptr< Feature >& feature )
            : edge( e ), frame( frame ), feature( feature ), is_deleted( false )
        {
        }
    };

    explicit BundleAdjustment( int32_t level, uint32_t numParameters );
    BundleAdjustment( const BundleAdjustment& rhs );
    BundleAdjustment( BundleAdjustment&& rhs );
    BundleAdjustment& operator=( const BundleAdjustment& rhs );
    BundleAdjustment& operator=( BundleAdjustment&& rhs );
    ~BundleAdjustment()       = default;

    double optimizePose( std::shared_ptr< Frame >& frame );
    double optimizeStructure( std::shared_ptr< Frame >& frame, const uint32_t maxNumberPoints );

    void twoViewBA( std::shared_ptr< Frame >& fstFrame,
                    std::shared_ptr< Frame >& secFrame,
                    double reprojectionError,
                    std::shared_ptr< Map >& map );

    void localBA( std::shared_ptr< Frame >& frame,
                  std::shared_ptr< Map >& map,
                  uint32_t& incoreectEdge1,
                  uint32_t& incorrectEdge2,
                  double& initError,
                  double& finalError );

    void globalBa( std::shared_ptr< Map >& map );

    void setupG2o( g2o::SparseOptimizer& optimizer );

    void runSparseBAOptimizer( g2o::SparseOptimizer& optimizer, uint32_t numIterations, double& initError, double& finalError );

    g2oFrameSE3* createG2oFrameSE3( const std::shared_ptr< Frame >& frame, const uint32_t id, const bool fixed );

    g2oPoint* createG2oPoint( const Eigen::Vector3d position, const uint32_t id, const bool fixed );

    g2oEdgeSE3* createG2oEdgeSE3(
      g2oFrameSE3* v_kf, g2oPoint* v_mp, const Eigen::Vector2d& up, bool robustKernel, double huberWidth, double weight = 1 );

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