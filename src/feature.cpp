#include "feature.hpp"

Feature::Feature( Frame& frame, const Eigen::Vector2d& feature, const uint8_t level )
    : m_frame( &frame )
    , m_type(FeatureType::EDGELET)
    , m_feature( feature )
    , m_bearingVec (m_frame->m_camera->inverseProject2d(feature))
    , m_level(level)
    , m_gradientOrientation( 0.0 )
    , m_gradientMagnitude( 1.0 )
    , m_point(nullptr)
{
}

Feature::Feature( Frame& frame,
                  const Eigen::Vector2d& feature,
                  const double gradientMagnitude,
                  const double gradientOrientation,
                  const uint8_t level )
    : m_frame( &frame )
    , m_type(FeatureType::EDGELET)
    , m_feature( feature )
    , m_bearingVec (m_frame->m_camera->inverseProject2d(feature))
    , m_level(level)
    , m_gradientOrientation( gradientOrientation )
    , m_gradientMagnitude( gradientMagnitude )
    , m_point(nullptr)
{
}