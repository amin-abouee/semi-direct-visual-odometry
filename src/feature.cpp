#include "feature.hpp"

uint64_t Feature::m_featureCounter;

Feature::Feature( const std::shared_ptr< Frame >& frame, const Eigen::Vector2d& feature, const uint8_t level )
    : m_id( m_featureCounter++ )
    , m_frame( frame )
    , m_type( FeatureType::EDGE )
    , m_pixelPosition( feature )
    , m_homogenous( feature.x(), feature.y(), 1.0 )
    , m_bearingVec( m_frame->m_camera->inverseProject2d( feature ) )
    , m_level( level )
    , m_gradientOrientation( 0.0 )
    , m_gradientMagnitude( 1.0 )
    , m_point( nullptr )
{
}

Feature::Feature( const std::shared_ptr< Frame >& frame,
                  const Eigen::Vector2d& feature,
                  const double gradientMagnitude,
                  const double gradientOrientation,
                  const uint8_t level )
    : m_id( m_featureCounter++ )
    , m_frame( frame )
    , m_type( FeatureType::EDGE )
    , m_pixelPosition( feature )
    , m_homogenous( feature.x(), feature.y(), 1.0 )
    , m_bearingVec( m_frame->m_camera->inverseProject2d( feature ) )
    , m_level( level )
    , m_gradientOrientation( gradientOrientation )
    , m_gradientMagnitude( gradientMagnitude )
    , m_point( nullptr )
{
}

void Feature::setPoint( std::shared_ptr< Point >& point )
{
    // http://www.cplusplus.com/reference/memory/shared_ptr/operator=/
    m_point = point;
    // https://stackoverflow.com/a/24130817/1804533
    assert( m_point.get() == point.get() );
}