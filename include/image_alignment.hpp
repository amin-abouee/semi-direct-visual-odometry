#ifndef __IMAGE_ALIGNMENT_HPP__
#define __IMAGE_ALIGNMENT_HPP__

#include <iostream>
#include <vector>

#include <Eigen/Core>
#include <opencv2/core.hpp>

#include "frame.hpp"

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

    Eigen::Matrix< double, Eigen::Dynamic, 6 > m_jacobian;
    cv::Mat m_refPatches;
    std::vector<bool> m_featureVisibility;

    void preCompute( Frame& frame, uint32_t level );
    void computeImageJac( Eigen::Matrix< double, 2, 6 >& imageJac,
                          const Eigen::Vector3d& point,
                          const double fx,
                          const double fy );
};

#endif /* __IMAGE_ALIGNMENT_HPP__ */