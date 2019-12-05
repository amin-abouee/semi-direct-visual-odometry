#include "pinhole_camera.hpp"
#include <iostream>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

PinholeCamera::PinholeCamera( double width,
                              double height,
                              double fx,
                              double fy,
                              double cx,
                              double cy,
                              double d0,
                              double d1,
                              double d2,
                              double d3,
                              double d4 )
    : m_width( width ), m_height( height )
{
    m_cvK          = ( cv::Mat_< double >( 3, 3 ) << fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0 );
    m_cvDistortion = ( cv::Mat_< float >( 1, 5 ) << d0, d1, d2, d3, d4 );
    m_distortion << d0, d1, d2, d3, d4;
    m_K << fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0;
    m_invK << 1 / fx, 0.0, -cx / fx, 0.0, 1 / fy, -cy / fy, 0.0, 0.0, 1.0;
    cv::initUndistortRectifyMap( m_cvK, m_cvDistortion, cv::Mat_< double >::eye( 3, 3 ), m_cvK,
                                 cv::Size( static_cast<int>(width), static_cast<int>(height) ), CV_16SC2, undistortedMapX, undistortedMapY );
    m_applyDistortion = std::fabs( d0 ) > 1e-5 ? true : false;
}


// http://ninghang.blogspot.com/2012/11/list-of-mat-type-in-opencv.html
PinholeCamera::PinholeCamera( const double width,
                              const double height,
                              const cv::Mat& cameraMatrix,
                              const cv::Mat& distortionCoeffs ) : m_width( width ), m_height( height )
{
    m_cvK = cameraMatrix.clone();
    m_cvDistortion = distortionCoeffs.clone();
    // std::cout << "cv K type: " << m_cvK.type() << std::endl;
    double* calib = m_cvK.ptr<double>(0);
    double* distro = m_cvDistortion.ptr<double>(0);
    m_distortion <<  distro[0], distro[1], distro[2], distro[3], distro[4];
    m_K << calib[0], calib[1], calib[2], calib[3], calib[4], calib[5], calib[6], calib[7], calib[8];
    m_invK << 1 / calib[0], 0.0, - calib[2] / calib[0], 0.0, 1 / calib[4], - calib[5] / calib[4], 0.0, 0.0, 1.0;
    cv::initUndistortRectifyMap( m_cvK, m_cvDistortion, cv::Mat_< double >::eye( 3, 3 ), m_cvK,
                                 cv::Size( static_cast<int>(width), static_cast<int>(height) ), CV_16SC2, undistortedMapX, undistortedMapY );
    m_applyDistortion = std::fabs( distro[0] ) > 1e-5 ? true : false;

    // std::cout << "m_cvK: " << m_cvK << std::endl;
    // std::cout << "m_cvDistortion: " << m_cvDistortion << std::endl;
    // std::cout << "m_K: " << m_K << std::endl;
    // std::cout << "m_invK: " << m_invK << std::endl;
    // std::cout << "m_applyDistortion: " << m_applyDistortion << std::endl;
}

// PinholeCamera::PinholeCamera( const PinholeCamera& rhs )
// {
// }

// PinholeCamera::PinholeCamera( PinholeCamera&& rhs )
// {
// }

// PinholeCamera& PinholeCamera::operator=( const PinholeCamera& rhs )
// {
// }

// PinholeCamera& PinholeCamera::operator=( PinholeCamera&& rhs )
// {
// }

Eigen::Vector2d PinholeCamera::project2d( const double x, const double y, const double z ) const
{
    Eigen::Vector2d pointImage;
    if ( m_applyDistortion == false )
    {
        pointImage( 0 ) = fx() * ( x / z ) + cx();
        pointImage( 1 ) = fy() * ( y / z ) + cy();
    }
    else
    {
        // https://docs.opencv.org/4.1.1/d9/d0c/group__calib3d.html#ga7dfb72c9cf9780a347fbe3d1c47e5d5a
        const double xPrime    = x / z;
        const double yPrime    = y / z;
        const double r2        = xPrime * xPrime + yPrime * yPrime;
        const double r4        = r2 * r2;
        const double r6        = r4 * r2;
        const double nominator = 1 + m_distortion( 0 ) * r2 + m_distortion( 1 ) * r4 + m_distortion( 2 ) * r6;
        // const double dominator = 1 + m_distortion[3] * r2 + m_distortion[4] * r4 + m_distortion[5] * r6;
        const double xZegond = ( xPrime * nominator ) + ( 2 * xPrime * yPrime ) + ( r2 + 2 * xPrime * xPrime );
        const double yZegond = ( yPrime * nominator ) + ( 2 * yPrime * xPrime ) + ( r2 + 2 * yPrime * yPrime );
        pointImage( 0 )      = fx() * xZegond + cx();
        pointImage( 1 )      = fy() * yZegond + cy();
    }
    return pointImage;
}

Eigen::Vector2d PinholeCamera::project2d( const Eigen::Vector3d& pointCamera ) const
{
    return project2d( pointCamera.x(), pointCamera.y(), pointCamera.z() );
}

Eigen::Vector3d PinholeCamera::inverseProject2d( const double x, const double y ) const
{
    Eigen::Vector3d pointCamera;
    if ( m_applyDistortion == false )
    {
        pointCamera( 0 ) = ( x - cx() ) / fx();
        pointCamera( 1 ) = ( y - cy() ) / fy();
        pointCamera( 2 ) = 1.0;
    }
    else
    {
        cv::Point2d uv( x, y ), px;
        const cv::Mat src_pt( 1, 1, CV_64FC2, &uv.x );
        cv::Mat dst_pt( 1, 1, CV_64FC2, &px.x );
        cv::undistortPoints( src_pt, dst_pt, m_cvK, m_cvDistortion );
        pointCamera( 0 ) = px.x;
        pointCamera( 1 ) = px.y;
        pointCamera( 2 ) = 1.0;
    }
    return pointCamera.normalized();
}

Eigen::Vector3d PinholeCamera::inverseProject2d( const Eigen::Vector2d& pointImage ) const
{
    return inverseProject2d( pointImage.x(), pointImage.y() );
}

const Eigen::Matrix3d& PinholeCamera::K() const
{
    return m_K;
}

const Eigen::Matrix3d& PinholeCamera::invK() const
{
    return m_invK;
}

const cv::Mat& PinholeCamera::K_cv() const
{
    return m_cvK;
}

double PinholeCamera::fx() const
{
    return m_K( 0, 0 );
}

double PinholeCamera::fy() const
{
    return m_K( 1, 1 );
}

double PinholeCamera::cx() const
{
    return m_K( 0, 2 );
}

double PinholeCamera::cy() const
{
    return m_K( 1, 2 );
}

Eigen::Vector2d PinholeCamera::focalLength() const
{
    return Eigen::Vector2d( fx(), fy() );
}

Eigen::Vector2d PinholeCamera::principlePoint() const
{
    return Eigen::Vector2d( cx(), cy() );
}

double PinholeCamera::width() const
{
    return m_width;
}

double PinholeCamera::height() const
{
    return m_height;
}

bool PinholeCamera::isInFrame( const Eigen::Vector2d& imagePoint, const double boundary ) const
{
    // std::cout << "img width: " << width() << ", height: " << height() << ", boundary" << boundary << std::endl;
    // std::cout << "imagePoint.x() >= boundary: " << (imagePoint.x() >= boundary) << std::endl;
    // std::cout << "imagePoint.y() >= boundary: " << (imagePoint.y() >= boundary) << std::endl;
    // std::cout << "imagePoint.x() < m_width - boundary: " << (imagePoint.x() < m_width - boundary) << std::endl;
    // std::cout << "imagePoint.y() < m_height - boundary: " << (imagePoint.y() < m_height - boundary) << std::endl;
    if ( imagePoint.x() >= boundary && imagePoint.y() >= boundary && imagePoint.x() < m_width - boundary &&
         imagePoint.y() < m_height - boundary )
        return true;
    return false;
}
bool PinholeCamera::isInFrame( const Eigen::Vector2d& imagePoint, const uint8_t level, const double boundary ) const
{
    if ( imagePoint.x() >= boundary && imagePoint.y() >= boundary &&
         imagePoint.x() < m_width / ( 1 << level ) - boundary && imagePoint.y() < m_height / ( 1 << level ) - boundary )
        return true;
    return false;
}

void PinholeCamera::undistortImage( const cv::Mat& distorted, cv::Mat& undistorted )
{
    if ( m_applyDistortion == true )
        cv::remap( distorted, undistorted, undistortedMapX, undistortedMapY, cv::INTER_LINEAR );
    else
        undistorted = distorted.clone();
}