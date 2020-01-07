#include "image_alignment.hpp"
#include "algorithm.hpp"
#include "feature.hpp"

#include <sophus/se3.hpp>

ImageAlignment::ImageAlignment( uint32_t patchSize, uint32_t minLevel, uint32_t maxLevel )
    : m_patchSize( patchSize )
    , m_halfPatchSize( patchSize / 2 )
    , m_patchArea( patchSize * patchSize )
    , m_minLevel( minLevel )
    , m_maxLevel( maxLevel )
{
}

double ImageAlignment::align( Frame& refFrame, Frame& curFrame )
{
    if ( refFrame.numberObservation() == 0 )
        return 0;

    Sophus::SE3d relativePose = algorithm::computeRelativePose( refFrame, curFrame );
    for ( uint32_t level( m_maxLevel ); level >= m_minLevel; level-- )
    {
        std::cout << "bobo" << std::endl;
    }
}

void ImageAlignment::preCompute( Frame& refFrame, uint32_t level )
{
    const uint32_t border   = m_halfPatchSize + 1;
    const cv::Mat& refImage = refFrame.m_imagePyramid.getImageAtLevel( level );
    const uint32_t stride   = refImage.cols;
    const double scale      = 1.0 / ( 1 << level );
    const Eigen::Vector3d C = refFrame.cameraInWorld();
    for ( const auto& feature : refFrame.m_frameFeatures )
    {
        const double u      = feature->m_feature.x() * scale;
        const double v      = feature->m_feature.y() * scale;
        const uint32_t uInt = static_cast< uint32_t >( std::floor( u ) );
        const uint32_t vInt = static_cast< uint32_t >( std::floor( v ) );
        if ( feature->m_point == nullptr || uInt - border < 0 || vInt - border < 0 || uInt + border > refImage.cols ||
             vInt + border > refImage.rows )
            continue;
        
        Eigen::Matrix<double, 2, 6> imageJac;
    }
}

void ImageAlignment::computeImageJac(Eigen::Vector3d& point, const double fx, const double fy)
{
    
}