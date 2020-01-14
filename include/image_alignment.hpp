#ifndef __IMAGE_ALIGNMENT_HPP__
#define __IMAGE_ALIGNMENT_HPP__

#include <iostream>
#include <vector>

#include <Eigen/Core>
#include <opencv2/core.hpp>

#include "frame.hpp"
#include "nlls.hpp"

class ImageAlignment
{
public:
    explicit ImageAlignment( uint32_t patchSize, int32_t minLevel, int32_t maxLevel );
    ImageAlignment( const ImageAlignment& rhs );
    ImageAlignment( ImageAlignment&& rhs );
    ImageAlignment& operator=( const ImageAlignment& rhs );
    ImageAlignment& operator=( ImageAlignment&& rhs );
    ~ImageAlignment()       = default;

    double align( Frame& refFrame, Frame& curFrame );

private:
    uint32_t m_patchSize;
    uint32_t m_halfPatchSize;
    uint32_t m_patchArea;
    uint32_t m_currentLevel;
    int32_t m_minLevel;
    int32_t m_maxLevel;

    // Eigen::Matrix< double, Eigen::Dynamic, 6 > m_jacobian;
    // Eigen::VectorXd m_residual;
    NLLS m_optimizer;
    cv::Mat m_refPatches;
    std::vector< bool > m_refVisibility;
    std::vector< bool > m_curVisibility;

    void computeJacobian( Frame& frame, uint32_t level );
    uint32_t computeResiduals( Frame& refFrame, Frame& curFrame, uint32_t level, Sophus::SE3d& pose );
    void computeImageJac( Eigen::Matrix< double, 2, 6 >& imageJac, const Eigen::Vector3d& point, const double fx, const double fy );

    void update( Sophus::SE3d& pose, const Eigen::VectorXd& dx );
};

#endif /* __IMAGE_ALIGNMENT_HPP__ */