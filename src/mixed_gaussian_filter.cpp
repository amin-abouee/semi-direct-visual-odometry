#include "mixed_gaussian_filter.hpp"
#include <cmath>

uint64_t MixedGaussianFilter::m_frameCounter;
uint64_t MixedGaussianFilter::m_filterCounter;

MixedGaussianFilter::MixedGaussianFilter( const std::shared_ptr< Feature >& feature, const double depthMean, const double depthMin )
    : m_id( m_filterCounter++ )
    , m_frameId( m_frameCounter )
    , m_a( 10 )
    , m_b( 10 )
    // mu reprsents the mean of inverse depth
    , m_mu( 1.0 / depthMean )
    , m_maxDepth( 1.0 / depthMin )
    // https://sixsigmastudyguide.com/what-is-six-sigma/
    // https://de.m.wikipedia.org/wiki/Datei:6_Sigma_Normal_distribution.jpg
    // in normal distribution, the USL stands for Upper Specification Limit. Means max_val = 6 * sigma => sigma = max_val / 6 
    , m_sigma (m_maxDepth / 6)
    // variance = sigma^2
    , m_var( m_sigma * m_sigma )
    , m_validity (true)
{
    m_feature = feature;
}