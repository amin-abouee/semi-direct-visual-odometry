#include "point.hpp"
#include "feature.hpp"

uint32_t Point::m_pointCounter;

Point::Point( const Eigen::Vector3d& point3D )
    : m_id( m_pointCounter++ )
    , m_position( point3D )
    , isNormalEstimated( false )
    , m_type( PointType::UNKNOWN )
    , m_lastPublishedTS( 0 )
    , m_lastProjectedKFId( -1 )
    , m_failedProjection( 0 )
    , m_succeededProjection( 0 )
    , m_lastOptimizedTime( 0 )
    , m_optG2oPoint (nullptr)
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
    , m_optG2oPoint (nullptr)
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

const std::shared_ptr< Feature >& Point::findFeature( const std::shared_ptr< Frame >& frame ) const
{
    for (const auto& feature : frame->m_features)
    {
        if (feature->m_point != nullptr && feature->m_point.get() == this)
        {
            return feature;
        }
    }
}

std::shared_ptr< Feature >& Point::findFeature( const std::shared_ptr< Frame >& frame )
{
    for (auto& feature : frame->m_features)
    {
        if (feature->m_point != nullptr && feature->m_point.get() == this)
        {
            return feature;
        }
    }
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
    /*
            frame k-n                       frame k-3               frame k-2              frame k-1            frame k (new frame)
      +--------------------+          +--------------------+ +--------------------+ +--------------------+     +--------------------+
      |                    |          |                    | |                    | |                    |     |                    |
      | f\                 | -------  |      f\            | |        f|          | |           /f       |     |              f     |
      |   --\              | -------  |        \           | |         |          | |          /         |     |            /-      |
      |      --\           |          |         -\         | |         \          | |         /          |     |          /-        |
      +- --------\---------+          +-----------\--------+ +---------|----------+ +--------|-----------+     +--------/-----------+
                  --\                             \                    |                    /                        /-
                    --\                           -\                   |                   /                      /--
                      --\                          \                   \                  /                     /-
                         --\                        \                  |                /                    /-
                            --\                      -\                |               /                   /-
                               --\                     \               |              /                 /--
                                  --\                   \              |             /                /-
                                      --\                 -\            \            |              /-    a_0 = angle(k, k-1)
                                         -\                 \           |           /             /-      a_1 = angle(k, k-2)
                                           --\               \          |          /            /-        a_2 = angle(k, k-3)
                                              --\             \         |         /          /--          .
                                                 --\          -\        |        /         /-             .
                                                    --\        \        \       /        /-               .
                                                      --\         \      |     /       /-                 a_n = angle(k, k-n)
                                                          --\      -\    |    /     /--
                                                              --\     \   |  |    /-                      min_a = min(a_0, ..., a_n)
                                                                --\    \  \  /  /-                        if min_a < 60:
                                                                   --\ -\ | / /-                              accept
                                                                      --\\|//-                            else
                                                                          ---                                 reject
                                                                      3D point
     */

    // compute the 3d vector between the point and camera origin in world coordinate and normalized it
    Eigen::Vector3d diffObs( poseInWorld - m_position );
    diffObs.normalized();

    double maxCosAngle = 0;
    for ( auto& ft : m_features )
    {
        // compute 3d vectors between the point and other camera origins in world coordinate and normalized them
        Eigen::Vector3d dir( ft->m_frame->cameraInWorld() - m_position );
        dir.normalized();
        // cos(alpha) = a * b / ||a|| * ||b|| -> we have already normalized a and b -> cos(alpha) = a * b
        // range cos in dot product: -1 <= cos(alpha) <= 1 (https://chortle.ccsu.edu/VectorLessons/vch09/vch09_6.html)
        // because frames are very close and point is visible in all of them, we assume the range is between 0° and 90° -> 0 <= cos(alpha)
        // <= 1 look for the closest frame and regarding to this fact that cos(0) = 1 and cos(90) = 0, the maximum should be selected
        double cosAngle = diffObs.dot( dir );
        if ( cosAngle > maxCosAngle )
        {
            feature     = ft;
            maxCosAngle = cosAngle;
        }
    }

    if ( maxCosAngle < 0.5 )  // assume that observations larger than 60° are useless
    {
        return false;
    }
    else
    {
        return true;
    }
}
