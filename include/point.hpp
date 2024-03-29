#ifndef __POINT_HPP__
#define __POINT_HPP__

#include "frame.hpp"

#include <g2o/types/slam3d/vertex_pointxyz.h>
#include <Eigen/Core>

#include <iostream>
#include <memory>
#include <vector>

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
    PointType m_type;
    uint64_t m_lastPublishedTS;      //!< Timestamp of last publishing.
    uint64_t m_lastProjectedKFId;    //!< Flag for the reprojection: don't reproject a pt twice.
    uint32_t m_failedProjection;     //!< Number of failed reprojections. Used to assess the quality of the point.
    uint32_t m_succeededProjection;  //!< Number of succeeded reprojections. Used to assess the quality of the point.
    uint32_t m_lastOptimizedTime;    //!< Timestamp of last point optimization

    g2o::VertexPointXYZ* m_optG2oPoint;

    explicit Point( const Eigen::Vector3d& point3D );
    explicit Point( const Eigen::Vector3d& point3D, const std::shared_ptr< Feature >& feature );
    Point( const Point& rhs ) = delete;
    Point( Point&& rhs )      = delete;
    Point& operator=( const Point& rhs ) = delete;
    Point& operator=( Point&& rhs ) = delete;
    ~Point()                        = default;

    void addFeature( const std::shared_ptr< Feature >& feature );
    void removeFrame( std::shared_ptr< Frame >& frame );
    const std::shared_ptr< Feature >& findFeature( const std::shared_ptr< Frame >& frame ) const;
    std::shared_ptr< Feature >& findFeature( const std::shared_ptr< Frame >& frame );
    bool findFrame (const std::shared_ptr< Frame >& frame) const;
    void computeNormal();
    std::size_t numberObservation() const;
    bool getCloseViewObservation( const Eigen::Vector3d& poseInWorld, std::shared_ptr< Feature >& feature ) const;
};

#endif /* __POINT_HPP__ */