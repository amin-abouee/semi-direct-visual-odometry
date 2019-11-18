#ifndef __FEATURE_HPP__
#define __FEATURE_HPP__

#include <iostream>
#include <Eigen/Core>

#include "frame.hpp"

class Feature final
{
public:

    enum class FeatureType : uint32_t
    {
        CORNER,
        EDGELET
    }

    FeatureType m_type;
    Frame* m_frame;
    Eigen::Vector2d m_px;
    Eigen::Vector3d m_bearing_vec;
    uint8_t level;
    Eigen::Vector3d m_point;
    Eigen::Vector2d m_gradientOrientation;


    explicit Feature();
    Feature(const Feature& rhs);
    Feature(Feature&& rhs);
    Feature& operator=(const Feature& rhs);
    Feature& operator=(Feature&& rhs);
    ~Feature() = default;

private:

};

#endif /*__FEATURE_HPP__ */