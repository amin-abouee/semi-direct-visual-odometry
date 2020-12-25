#ifndef __IMAGE_ALIGNMENT_HPP__
#define __IMAGE_ALIGNMENT_HPP__

#include "algorithm.hpp"
#include "frame.hpp"
#include "optimizer.hpp"

#include <Eigen/Core>
#include <opencv2/core.hpp>

#include <iostream>
#include <memory>
#include <vector>

class ImageAlignment
{
public:
    explicit ImageAlignment( uint32_t patchSize, int32_t minLevel, int32_t maxLevel, uint32_t numParameters );
    ImageAlignment( const ImageAlignment& rhs );
    ImageAlignment( ImageAlignment&& rhs );
    ImageAlignment& operator=( const ImageAlignment& rhs );
    ImageAlignment& operator=( ImageAlignment&& rhs );
    ~ImageAlignment()       = default;

    double align( std::shared_ptr< Frame >& refFrame, std::shared_ptr< Frame >& curFrame );

    // Matrix<double, 6, 6> getFisherInformation();

private:
    uint32_t m_patchSize;
    uint32_t m_halfPatchSize;
    uint32_t m_patchArea;
    uint32_t m_currentLevel;
    int32_t m_minLevel;
    int32_t m_maxLevel;

    Optimizer m_optimizer;
    // cv::Mat m_refPatches;
    Eigen::MatrixXd m_refPatches;
    std::vector< bool > m_refVisibility;
    // std::vector< bool > m_curVisibility;

    void computeJacobian( const std::shared_ptr< Frame >& frame, const uint32_t level );
    
    bool computeJacobianSingleFeature( const std::shared_ptr< Feature >& feature,
                                       const algorithm::MapXRowConst& imageEigen,
                                       const int32_t border,
                                       const Eigen::Vector3d& cameraInWorld,
                                       const double scale,
                                       const double scaledFx,
                                       const double scaledFy,
                                       uint32_t& cntFeature );

    void computeImageJac( Eigen::Matrix< double, 2, 6 >& imageJac, const Eigen::Vector3d& point, const double fx, const double fy );

    uint32_t computeResiduals( const std::shared_ptr< Frame >& refFrame,
                               const std::shared_ptr< Frame >& curFrame,
                               const uint32_t level,
                               const Sophus::SE3d& pose );

    bool computeResidualSingleFeature( const std::shared_ptr< Feature >& feature,
                                       const algorithm::MapXRowConst& imageEigen,
                                       const std::shared_ptr< Frame >& curFrame,
                                       const Sophus::SE3d& pose,
                                       const int32_t border,
                                       const Eigen::Vector3d& cameraInWorld,
                                       const double scale,
                                       uint32_t& cntFeature,
                                       uint32_t& cntTotalProjectedPixels );

    void update( Sophus::SE3d& pose, const Eigen::VectorXd& dx );
    void resetParameters();
};

#endif /* __IMAGE_ALIGNMENT_HPP__ */