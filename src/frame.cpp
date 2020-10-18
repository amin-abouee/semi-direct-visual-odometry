#include "frame.hpp"
#include "feature.hpp"

uint64_t Frame::m_frameCounter;

Frame::Frame( const std::shared_ptr<PinholeCamera>& camera, const cv::Mat& img, const uint32_t maxImagePyramid, const double timestamp )
    : m_id( m_frameCounter++ )
    , m_camera( camera )
    , m_TransW2F( Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero() )
    , m_imagePyramid( img, maxImagePyramid )
    , m_keyFrame( false )
    , m_timestamp ( timestamp )
{
}

void Frame::initFrame( const cv::Mat& img, const uint32_t maxImagePyramid )
{
    if ( img.empty() || img.type() != CV_8UC1 || img.cols != m_camera->width() || img.rows != m_camera->height() )
        throw std::runtime_error( "IMAGE CORROUPTED" );

    m_imagePyramid.createImagePyramid( img, maxImagePyramid );
}

void Frame::setKeyframe()
{
    m_keyFrame = true;
}

void Frame::addFeature( std::shared_ptr< Feature >& feature )
{
    m_frameFeatures.emplace_back( feature );
}

void Frame::removeFeature( std::shared_ptr< Feature >& feature )
{
    auto find = [&feature]( std::shared_ptr< Feature >& f ) -> bool {
        //TODO: check with get() and raw pointer
        if ( f == feature )
            return true;
        return false;
    };
    auto element = std::remove_if( m_frameFeatures.begin(), m_frameFeatures.end(), find );
    m_frameFeatures.erase(element, m_frameFeatures.end());
}

std::size_t Frame::numberObservation() const
{
    return m_frameFeatures.size();
}

bool Frame::isVisible( const Eigen::Vector3d& point3D ) const
{
    const Eigen::Vector3d cameraPoint = m_TransW2F * point3D;
    if ( cameraPoint.z() < 0.0 )
        return false;
    const Eigen::Vector2d imagePoint = camera2image( cameraPoint );
    return m_camera->isInFrame( imagePoint );
}

bool Frame::isKeyframe() const
{
    return m_keyFrame;
}

Eigen::Vector2d Frame::world2image( const Eigen::Vector3d& point3D_w ) const
{
    const Eigen::Vector3d cameraPoint = world2camera( point3D_w );
    return camera2image( cameraPoint );
}

Eigen::Vector3d Frame::world2camera( const Eigen::Vector3d& point3D_w ) const
{
    return m_TransW2F * point3D_w;
}

Eigen::Vector3d Frame::camera2world( const Eigen::Vector3d& point3D_c ) const
{
    return m_TransW2F.inverse() * point3D_c;
}

Eigen::Vector2d Frame::camera2image( const Eigen::Vector3d& point3D_c ) const
{
    return m_camera->project2d( point3D_c );
}

Eigen::Vector3d Frame::image2camera( const Eigen::Vector2d& point2D, const double depth ) const
{
    return m_camera->inverseProject2d( point2D ) * depth;
}

Eigen::Vector3d Frame::image2world( const Eigen::Vector2d& point2D, const double depth ) const
{
    const Eigen::Vector3d cameraPoint = image2camera( point2D, depth );
    return camera2world( cameraPoint );
}

/// compute position of camera in world cordinate C = -R^t * t
Eigen::Vector3d Frame::cameraInWorld() const
{
    // return m_TransW2F.inverse().translation();
    return -m_TransW2F.rotationMatrix().transpose() * m_TransW2F.translation();
}