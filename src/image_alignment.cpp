#include "image_alignment.hpp"

ImageAlignment::ImageAlignment( uint32_t patchSize, uint32_t minLevel, uint32_t maxLevel )
    : m_patchSize( patchSize )
    , m_halfPatchSize( patchSize / 2 )
    , m_patchArea( patchSize * patchSize )
    , m_minLevel( minLevel )
    , m_maxLevel( maxLevel )
{
}

double ImageAlignment::solve( Frame& refFrame, Frame& curFrame )
{
    if ( refFrame.numberObservation() == 0 )
        return 0;
}