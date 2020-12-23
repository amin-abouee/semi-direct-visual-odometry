#include "image_pyramid.hpp"

#define SIMD_OPENCV_ENABLE
#include <Simd/SimdLib.hpp>
#include <Simd/SimdView.hpp>
#include <easylogging++.h>

#define Feature_Log( LEVEL ) CLOG( LEVEL, "Feature" )

ImagePyramid::ImagePyramid( const std::size_t level )
{
    m_vecImages.reserve( level );
}

ImagePyramid::ImagePyramid( const cv::Mat& baseImage, const std::size_t level )
{
    cv::Mat gradientImage ( baseImage.size(), CV_8U );
    Simd::View< Simd::Allocator > src = baseImage;
    Simd::View< Simd::Allocator > dst = gradientImage;
    Simd::AbsGradientSaturatedSum( src, dst );

    m_vecImages.reserve( level );
    m_vecGradientImages.reserve (level);
    cv::Mat resizeImage( baseImage );
    cv::Mat resizeGradientImage( gradientImage );

    for ( std::size_t i( 0 ); i < level; i++ )
    {
        m_vecImages.emplace_back( resizeImage );
        m_vecGradientImages.emplace_back( resizeGradientImage );
        cv::pyrDown( resizeImage, resizeImage );
        cv::pyrDown( resizeGradientImage, resizeGradientImage );
    }
}

void ImagePyramid::createImagePyramid( const cv::Mat& baseImage, const std::size_t level )
{
    cv::Mat gradientImage ( baseImage.size(), CV_8U );
    Simd::View< Simd::Allocator > src = baseImage;
    Simd::View< Simd::Allocator > dst = gradientImage;
    Simd::AbsGradientSaturatedSum( src, dst );

    cv::Mat resizeImage( baseImage );
    cv::Mat resizeGradientImage( gradientImage );
    for ( std::size_t i( 0 ); i < level; i++ )
    {
        m_vecImages.emplace_back( resizeImage );
        m_vecGradientImages.emplace_back( resizeGradientImage );
        cv::pyrDown( resizeImage, resizeImage );
        cv::pyrDown( resizeGradientImage, resizeGradientImage );
    }
}

const std::vector< cv::Mat >& ImagePyramid::getAllImages() const
{
    return m_vecImages;
}

std::vector< cv::Mat >& ImagePyramid::getAllImages()
{
    return m_vecImages;
}

const cv::Mat& ImagePyramid::getImageAtLevel( const std::size_t level ) const
{
    // if ( level < m_vecImages.size() )
    return m_vecImages[ level ];
}

cv::Mat& ImagePyramid::getImageAtLevel( const std::size_t level )
{
    // if ( level < m_vecImages.size() )
    return m_vecImages[ level ];
}

const cv::Mat& ImagePyramid::getBaseImage() const
{
    // if ( m_vecImages.size() > 0 )
    return m_vecImages[ 0 ];
}

cv::Mat& ImagePyramid::getBaseImage()
{
    // if ( m_vecImages.size() > 0 )
    return m_vecImages[ 0 ];
}

const cv::Mat& ImagePyramid::getBaseGradientImage() const
{
    return m_vecGradientImages [0];
}

cv::Mat& ImagePyramid::getBaseGradientImage()
{
    return m_vecGradientImages [0];
}

std::size_t ImagePyramid::getSizeImagePyramid() const
{
    return m_vecImages.size();
}

cv::Size ImagePyramid::getImageSizeAtLevel( const std::size_t level )
{
    if ( level < m_vecImages.size() )
        return m_vecImages[ level ].size();
    else
        return ( cv::Size( 0, 0 ) );
}

cv::Size ImagePyramid::getBaseImageSize() const
{
    return ( getBaseImage().size() );
}

void ImagePyramid::clear()
{
    m_vecImages.clear();
    m_baseImageWidth  = 0;
    m_baseImageHeight = 0;
}
