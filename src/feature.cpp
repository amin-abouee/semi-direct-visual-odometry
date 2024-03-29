#include "feature.hpp"

uint64_t Feature::m_featureCounter;

Feature::Feature( const std::shared_ptr< Frame >& frame,
                  const Eigen::Vector2d& pixelPosition,
                  const uint8_t level,
                  const FeatureType& type )
    : m_id( m_featureCounter++ )
    , m_frame( frame )
    , m_type( type )
    , m_pixelPosition( pixelPosition )
    , m_homogenous( pixelPosition.x(), pixelPosition.y(), 1.0 )
    , m_bearingVec( m_frame->m_camera->inverseProject2d( pixelPosition ) )
    , m_gradientMagnitude( 1.0 )
    , m_gradientOrientation( 0.0 )
    , m_level( level )
    , m_point( nullptr )
{
}

Feature::Feature( const std::shared_ptr< Frame >& frame,
                  const Eigen::Vector2d& pixelPosition,
                  const double gradientMagnitude,
                  const double gradientOrientation,
                  const uint8_t level,
                  const FeatureType& type )
    : m_id( m_featureCounter++ )
    , m_frame( frame )
    , m_type( type )
    , m_pixelPosition( pixelPosition )
    , m_homogenous( pixelPosition.x(), pixelPosition.y(), 1.0 )
    , m_bearingVec( m_frame->m_camera->inverseProject2d( pixelPosition ) )
    , m_gradientMagnitude( gradientMagnitude )
    , m_gradientOrientation( gradientOrientation )
    , m_level( level )
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