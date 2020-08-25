#include "point.hpp"
#include "feature.hpp"

uint32_t Point::m_pointCounter;

Point::Point( const Eigen::Vector3d& point3D )
    : m_id( m_pointCounter++ )
    , m_position( point3D )
    , isNormalEstimated( false )
    // , m_numSuccessProjection( 0 )
    , m_type( PointType::UNKNOWN )
{
}

Point::Point(const Eigen::Vector3d& point3D, const std::shared_ptr<Feature>& feature)
    : m_id( m_pointCounter++ )
    , m_position( point3D )
    , isNormalEstimated( false )
    , m_type( PointType::UNKNOWN )
{
    m_features.push_back(feature);
}

void Point::addFeature(const std::shared_ptr<Feature>& feature)
{
    m_features.push_back(feature);
}

void Point::removeFrame (std::shared_ptr< Frame >& frame)
{
    auto find = [ &frame ]( std::shared_ptr< Feature >& feature ) -> bool {
        if ( feature->m_frame == frame )
            return true;
        return false;
    };
    auto element = std::remove_if( m_features.begin(), m_features.end(), find );
    m_features.erase( element, m_features.end() );
}

void Point::computeNormal()
{

}

std::size_t Point::numberObservation() const
{
    return m_features.size();
}
