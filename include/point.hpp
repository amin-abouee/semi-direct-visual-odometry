#ifndef __POINT_HPP__
#define __POINT_HPP__

#include <iostream>
#include <memory>
#include <vector>

#include "frame.hpp"

#include <Eigen/Core>

class Feature;
class Point final
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    enum class PointType : uint32_t
    {
        GOOD,
        DELETED,
        CANDIDATE,
        UNKNOWN
    };

    static uint32_t m_pointCounter;
    uint32_t m_id;
    Eigen::Vector3d m_position;
    Eigen::Vector3d m_normal;
    Eigen::Matrix3d m_normalCov;
    bool isNormalEstimated;
    std::vector< std::shared_ptr< Feature > > m_features;
    // uint32_t m_numSuccessProjection;
    PointType m_type;

    explicit Point( const Eigen::Vector3d& point3D );
    explicit Point( const Eigen::Vector3d& point3D, const std::shared_ptr< Feature >& feature );
    Point( const Point& rhs ) = delete;
    Point( Point&& rhs )      = delete;
    Point& operator=( const Point& rhs ) = delete;
    Point& operator=( Point&& rhs ) = delete;
    ~Point()                        = default;

    void addFeature( const std::shared_ptr< Feature >& feature );
    void removeFrame (std::shared_ptr< Frame >& frame);
    std::shared_ptr< Feature >& findFeature( std::shared_ptr< Frame >& frame );
    void computeNormal();
    std::size_t numberObservation() const;
};

#endif /* __POINT_HPP__ */