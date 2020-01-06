#ifndef __IMAGE_ALIGNMENT_HPP__
#define __IMAGE_ALIGNMENT_HPP__

#include <iostream>
#include <vector>

#include <Eigen/Core>

#include "frame.hpp"

class ImageAlignment
{
public:
    explicit ImageAlignment( uint32_t halfPatchSize,
                             uint32_t patchSize,
                             uint32_t patchArea,
                             uint32_t minLevel,
                             uint32_t maxLevel );
    ImageAlignment( const ImageAlignment& rhs );
    ImageAlignment( ImageAlignment&& rhs );
    ImageAlignment& operator=( const ImageAlignment& rhs );
    ImageAlignment& operator=( ImageAlignment&& rhs );
    ~ImageAlignment()       = default;

    double solve (Frame& refFrame, Frame& curFrame);

private:
    uint32_t m_halfPatchSize;
    uint32_t m_patchSize;
    uint32_t m_patchArea;
    uint32_t m_currentLevel;
    uint32_t m_minLevel;
    uint32_t m_maxLevel;
};

#endif /* __IMAGE_ALIGNMENT_HPP__ */