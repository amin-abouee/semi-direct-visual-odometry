#include "image_alignment.hpp"

ImageAlignment::ImageAlignment(
  uint32_t halfPatchSize, uint32_t patchSize, uint32_t patchArea, uint32_t minLevel, uint32_t maxLevel )
    : m_halfPatchSize( halfPatchSize )
    , m_patchSize( patchSize )
    , m_patchArea( patchArea )
    , m_minLevel( minLevel )
    , m_maxLevel( maxLevel )
{

}

double ImageAlignment::solve (Frame& refFrame, Frame& curFrame)
{
    if (refFrame.numberObservation() == 0)
        return 0
    
}