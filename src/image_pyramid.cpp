#include "image_pyramid.hpp"

#include "easylogging++.h"
#define Feature_Log( LEVEL ) CLOG( LEVEL, "Feature" )

ImagePyramid::ImagePyramid( const std::size_t level )
{
    m_vecImages.reserve( level );
}

ImagePyramid::ImagePyramid( const cv::Mat& baseImage, const std::size_t level )
{
    m_vecImages.reserve( level );
    cv::Mat resizeImage( baseImage );
    for ( std::size_t i( 0 ); i < level; i++ )
    {
        m_vecImages.emplace_back( resizeImage );
        cv::pyrDown( resizeImage, resizeImage );
    }
}

void ImagePyramid::createImagePyramid( const cv::Mat& baseImage, const std::size_t level )
{
    cv::Mat resizeImage( baseImage );
    for ( std::size_t i( 0 ); i < level; i++ )
    {
        m_vecImages.emplace_back( resizeImage );
        cv::pyrDown( resizeImage, resizeImage );
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
