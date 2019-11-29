#ifndef __POINT_HPP__
#define __POINT_HPP__

#include <iostream>

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
    }

    static uint32_t m_frameCounter;
    uint32_t m_id;
    Eigen::Vector3d m_point;
    Eigen::Vector3d m_normal;
    Eigen::Matrix3d m_normalCov;
    std::Vector< Feature* > m_features;
    uint32_t m_numObservation;
    PointType m_type;

    explicit Point();
    Point( const Point& rhs ) = delete;
    Point( Point&& rhs )      = delete;
    Point& operator=( const Point& rhs ) = delete;
    Point& operator=( Point&& rhs ) = delete;
    ~Point()                        = default;
}

#endif /* __POINT_HPP__ */