#include "point.hpp"

uint32_t Point::m_pointCounter;

Point::Point( const Eigen::Vector3d& point3D )
    : m_id( m_pointCounter++ )
    , m_position( point3D )
    , isNormalEstimated( false )
    , m_numSuccessProjection( 0 )
    , m_type( PointType::UNKNOWN )
{
}

Point::Point(const Eigen::Vector3d& point3D, Feature* feature)
    : m_id( m_pointCounter++ )
    , m_position( point3D )
    , isNormalEstimated( false )
    , m_numSuccessProjection( 1 )
    , m_type( PointType::UNKNOWN )
{
    m_features.push_back(feature);
}

void Point::addFeature(Feature* feature)
{
    m_features.push_back(feature);
    m_numSuccessProjection++;
}

void Point::computeNormal()
{

}

std::size_t Point::numberObservation() const
{
    return m_numSuccessProjection;
}
