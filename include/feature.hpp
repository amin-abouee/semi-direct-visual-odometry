#ifndef __FEATURE_HPP__
#define __FEATURE_HPP__

#include <functional>
#include <iostream>
#include <memory>

#include <Eigen/Core>
#include <iostream>
#include <memory>

#include "frame.hpp"
#include "point.hpp"

class Feature final
{
public:
EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    enum class FeatureType : uint32_t
    {
        CORNER,
        EDGELET
    };

    static uint64_t m_featureCounter;
    uint64_t m_id;
    std::shared_ptr<Frame> m_frame;
    FeatureType m_type;
    Eigen::Vector2d m_feature;
    Eigen::Vector3d m_homogenous;
    Eigen::Vector3d m_bearingVec;
    uint8_t m_level;
    double m_gradientOrientation;
    double m_gradientMagnitude;
    std::shared_ptr<Point> m_point;

    explicit Feature( const std::shared_ptr<Frame>& frame, const Eigen::Vector2d& feature, const uint8_t level );
    explicit Feature( const std::shared_ptr<Frame>& frame,
                      const Eigen::Vector2d& feature,
                      const double gradientMagnitude,
                      const double m_gradientOrientation,
                      const uint8_t level );
    Feature( const Feature& rhs );
    Feature( Feature&& rhs );
    Feature& operator=( const Feature& rhs );
    Feature& operator=( Feature&& rhs );
    // ~Feature()       = default;

    void setPoint(std::shared_ptr<Point>& point);

private:
};

#endif /*__FEATURE_HPP__ */