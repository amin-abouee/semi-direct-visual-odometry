#ifndef __FEATURE_HPP__
#define __FEATURE_HPP__

#include <functional>
#include <iostream>

#include <Eigen/Core>
#include <iostream>

#include "frame.hpp"

class Feature final
{
public:
    enum class FeatureType : uint32_t
    {
        CORNER,
        EDGELET
    };

    Frame* m_frame;
    FeatureType m_type;
    Eigen::Vector2d m_feature;
    Eigen::Vector3d m_homogenous;
    Eigen::Vector3d m_bearingVec;
    uint8_t m_level;
    double m_gradientOrientation;
    double m_gradientMagnitude;
    Point* m_point;

    explicit Feature( Frame& frame, const Eigen::Vector2d& feature, const uint8_t level );
    explicit Feature( Frame& frame,
                      const Eigen::Vector2d& feature,
                      const double gradientMagnitude,
                      const double m_gradientOrientation,
                      const uint8_t level );
    Feature( const Feature& rhs );
    Feature( Feature&& rhs );
    Feature& operator=( const Feature& rhs );
    Feature& operator=( Feature&& rhs );
    // ~Feature()       = default;

private:
};

#endif /*__FEATURE_HPP__ */