#include "frame.hpp"
#include "feature.hpp"

uint64_t Frame::m_frameCounter = 0;

Frame::Frame( const std::shared_ptr< PinholeCamera >& camera,
              const cv::Mat& img,
              const uint32_t maxImagePyramid,
              const uint64_t timestamp,
              const std::shared_ptr< Frame > lastKeyframe )
    : m_id( m_frameCounter++ )
    , m_camera( camera )
    , m_absPose( Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero() )
    , m_imagePyramid( maxImagePyramid )
    , m_keyFrame( false )
    , m_timestamp( timestamp )
    , m_lastKeyframe( lastKeyframe )
    , m_optG2oFrame (nullptr)
{
    if ( img.empty() || img.type() != CV_8UC1 || img.cols != static_cast< int32_t >( m_camera->width() ) ||
         img.rows != static_cast< int32_t >( m_camera->height() ) )
    {
        throw std::runtime_error( "Image Corrupted" );
    }

    m_imagePyramid.createImagePyramid( img, maxImagePyramid );
}

void Frame::setKeyframe()
{
    m_keyFrame = true;
}

void Frame::addFeature( std::shared_ptr< Feature >& feature )
{
    m_features.emplace_back( feature );
}

void Frame::removeFeature( std::shared_ptr< Feature >& feature )
{
    auto find = [ &feature ]( std::shared_ptr< Feature >& f ) -> bool {
        // TODO: check with get() and raw pointer
        if ( f == feature )
            return true;
        return false;
    };
    auto element = std::remove_if( m_features.begin(), m_features.end(), find );
    m_features.erase( element, m_features.end() );
}

std::size_t Frame::numberObservation() const
{
    return m_features.size();
}

bool Frame::isVisible( const Eigen::Vector3d& point3d_w ) const
{
    const Eigen::Vector3d cameraPoint = m_absPose * point3d_w;
    if ( cameraPoint.z() < 0.0 )
        return false;
    const Eigen::Vector2d imagePoint = camera2image( cameraPoint );
    return m_camera->isInFrame( imagePoint );
}

bool Frame::isKeyframe() const
{
    return m_keyFrame;
}

Eigen::Vector2d Frame::world2image( const Eigen::Vector3d& point3d_w ) const
{
    const Eigen::Vector3d cameraPoint = world2camera( point3d_w );
    return camera2image( cameraPoint );
}

Eigen::Vector3d Frame::world2camera( const Eigen::Vector3d& point3d_w ) const
{
    return m_absPose * point3d_w;
}

Eigen::Vector3d Frame::camera2world( const Eigen::Vector3d& point3d_c ) const
{
    return m_absPose.inverse() * point3d_c;
}

Eigen::Vector2d Frame::camera2image( const Eigen::Vector3d& point3d_c ) const
{
    return m_camera->project2d( point3d_c );
}

Eigen::Vector3d Frame::image2camera( const Eigen::Vector2d& pixelPosition, const double depth ) const
{
    return m_camera->inverseProject2d( pixelPosition ) * depth;
}

Eigen::Vector3d Frame::image2world( const Eigen::Vector2d& pixelPosition, const double depth ) const
{
    const Eigen::Vector3d cameraPoint = image2camera( pixelPosition, depth );
    return camera2world( cameraPoint );
}

/// compute position of camera in world cordinate C = -R^t * t
Eigen::Vector3d Frame::cameraInWorld() const
{
    // return m_absPose.inverse().translation();
    return -m_absPose.rotationMatrix().transpose() * m_absPose.translation();
}