#include "point.hpp"
#include "feature.hpp"

uint32_t Point::m_pointCounter;

Point::Point( const Eigen::Vector3d& point3D )
    : m_id( m_pointCounter++ )
    , m_position( point3D )
    , isNormalEstimated( false )
    // , m_numSuccessProjection( 0 )
    , m_type( PointType::UNKNOWN )
    , m_lastPublishedTS( 0 )
    , m_lastProjectedKFId( -1 )
    , m_failedProjection( 0 )
    , m_succeededProjection( 0 )
    , m_lastOptimizedTime( 0 )

{
}

Point::Point( const Eigen::Vector3d& point3D, const std::shared_ptr< Feature >& feature )
    : m_id( m_pointCounter++ )
    , m_position( point3D )
    , isNormalEstimated( false )
    , m_type( PointType::UNKNOWN )
    , m_lastPublishedTS( 0 )
    , m_lastProjectedKFId( -1 )
    , m_failedProjection( 0 )
    , m_succeededProjection( 0 )
    , m_lastOptimizedTime( 0 )
{
    m_features.push_back( feature );
}

void Point::addFeature( const std::shared_ptr< Feature >& feature )
{
    m_features.push_back( feature );
}

void Point::removeFrame( std::shared_ptr< Frame >& frame )
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
    const auto& feature = m_features.back();
    m_normal            = feature->m_frame->m_absPose.rotationMatrix().transpose() * ( -feature->m_bearingVec );
    m_normalCov =
      Eigen::DiagonalMatrix< double, 3, 3 >( std::pow( 20 / ( m_position - feature->m_frame->cameraInWorld() ).norm(), 2 ), 1.0, 1.0 );
    isNormalEstimated = true;
}

std::size_t Point::numberObservation() const
{
    return m_features.size();
}

bool Point::getCloseViewObservation( const Eigen::Vector3d& poseInWorld, std::shared_ptr< Feature >& feature ) const
{
    Eigen::Vector3d diffObs( poseInWorld - m_position );
    diffObs.normalized();
    double minCosAngle = 0;
    for ( auto& ft : m_features )
    {
        Eigen::Vector3d dir( ft->m_frame->cameraInWorld() - m_position );
        dir.normalized();
        double cosAngle = diffObs.dot( dir );
        if ( cosAngle < minCosAngle )
        {
            feature     = ft;
            minCosAngle = cosAngle;
        }
    }

    if ( minCosAngle < 0.5 )  // assume that observations larger than 60° are useless
    {
        return false;
    }
    else
    {
        return true;
    }
}
