#ifndef  FEATURE_HPP
#define  FEATURE_HPP

#include "frame.hpp"
#include "point.hpp"

#include <Eigen/Core>

#include <functional>
#include <iostream>
#include <memory>

/// @brief Feature information, is tracked across frames.
class Feature final
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    /// determine the type of feature
    enum class FeatureType : uint32_t
    {
        CORNER,
        EDGE,
        DEFAULT
    };

    static uint64_t m_featureCounter;  ///< Counts the number of created features. Used to set the unique id
    uint64_t m_id;                     ///< Unique id of the feature
    std::shared_ptr< Frame > m_frame;  ///< Owner frame in which the feature was detected
    FeatureType m_type;                ///< Type of this feature
    Eigen::Vector2d m_pixelPosition;   ///< Pixel coordinates on pyramid level 0
    Eigen::Vector3d m_homogenous;      ///< Location in homogenous coordinate (x, y, 1)
    Eigen::Vector3d
      m_bearingVec;  ///< Unit-bearing vector of the feature. K.inverse() * (x, y, 1) -> point in camera coordinate but up to scale
    double m_gradientMagnitude;        ///< Gradient magnitude is computed as: sqrt(dx * dx + dy * dy)
    double m_gradientOrientation;      ///< Gradient orientation defined as angle: arctan (dx, dy)
    uint8_t m_level;                   ///< Image pyramid level where feature was extracted
    std::shared_ptr< Point > m_point;  ///< 3D point in world coordinate which corresponds to the feature

    /// @brief Construct a new Feature object
    ///
    /// @param[in] frame Owner frame
    /// @param[in] pixelPosition 2D position in pixel coordinate (x, y)
    /// @param[in] level Image pyramid level of detected point
    /// @param[in] type Feature type
    explicit Feature( const std::shared_ptr< Frame >& frame,
                      const Eigen::Vector2d& pixelPosition,
                      const uint8_t level,
                      const FeatureType& type = FeatureType::DEFAULT );

    /// @brief Construct a new Feature object
    ///
    /// @param[in] frame Owner frame
    /// @param[in] pixelPosition 2D position in pixel coordinate (x, y)
    /// @param[in] gradientMagnitude Gradient magnitude
    /// @param[in] gradientOrientation Gradient orientation defined as angle
    /// @param[in] level Image pyramid level of detected point
    /// @param[in] type Feature type
    explicit Feature( const std::shared_ptr< Frame >& frame,
                      const Eigen::Vector2d& pixelPosition,
                      const double gradientMagnitude,
                      const double gradientOrientation,
                      const uint8_t level,
                      const FeatureType& type = FeatureType::DEFAULT );

    Feature( const Feature& rhs ) = delete;
    Feature( Feature&& rhs )      = delete;
    Feature& operator=( const Feature& rhs ) = delete;
    Feature& operator=( Feature&& rhs ) = delete;
    ~Feature()                          = default;

    /// @brief Set the Point object
    ///
    /// @param[in] point 3D point in world coordinate which corresponds to the feature
    void setPoint( std::shared_ptr< Point >& point );

private:
};

#endif /* FEATURE_HPP */