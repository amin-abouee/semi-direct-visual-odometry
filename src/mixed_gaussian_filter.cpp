#include "mixed_gaussian_filter.hpp"
#include <cmath>

uint64_t MixedGaussianFilter::m_frameCounter;
uint64_t MixedGaussianFilter::m_filterCounter;

MixedGaussianFilter::MixedGaussianFilter( const std::shared_ptr< Feature >& feature, const double depthMean, const double depthMin )
    : m_id( m_filterCounter++ )
    , m_frameId( m_frameCounter )
    , m_a( 10 )
    , m_b( 10 )
    , m_mu( 1.0 / depthMean )
    , m_maxDepth( 1.0 / depthMin )
    , m_var( m_maxDepth * m_maxDepth / 36 )
    , m_sigma (std::sqrt(m_var))
    , m_validity (true)
{
    m_feature = feature;
}