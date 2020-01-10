#include "image_alignment.hpp"
#include "algorithm.hpp"
#include "feature.hpp"

#include <sophus/se3.hpp>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

ImageAlignment::ImageAlignment( uint32_t patchSize, int32_t minLevel, int32_t maxLevel )
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

    const std::size_t numObservations = refFrame.numberObservation();
    m_refPatches                      = cv::Mat( numObservations, m_patchArea, CV_32F );
    m_jacobian.resize( numObservations * m_patchArea, Eigen::NoChange );
    m_residual.resize(numObservations);
    m_featureVisibility.resize( numObservations, false );

    std::cout << "Reference Patch size: " << m_refPatches.size << std::endl;

    Sophus::SE3d relativePose = algorithm::computeRelativePose( refFrame, curFrame );
    // when we wanna compare a uint32 with an int32, the c++ can not compare -1 with 0
    for ( int32_t level( m_maxLevel ); level >= m_minLevel; level-- )
    {
        computeJacobian( refFrame, level );
    }
    return 1.0;
}

void ImageAlignment::computeJacobian( Frame& frame, uint32_t level )
{
    const int32_t border    = m_halfPatchSize + 2;
    const cv::Mat& refImage = frame.m_imagePyramid.getImageAtLevel( level );
    const algorithm::MapXRowConst refImageEigen( refImage.ptr< float >(), refImage.rows, refImage.cols );
    const uint32_t stride       = refImage.cols;
    const double levelDominator = 1 << level;
    const double scale          = 1.0 / levelDominator;
    const Eigen::Vector3d C     = frame.cameraInWorld();
    const double fx             = frame.m_camera->fx() / levelDominator;
    const double fy             = frame.m_camera->fy() / levelDominator;
    uint32_t cntFeature         = 0;
    for ( const auto& feature : frame.m_frameFeatures )
    {
        const double u     = feature->m_feature.x() * scale;
        const double v     = feature->m_feature.y() * scale;
        const int32_t uInt = static_cast< int32_t >( std::floor( u ) );
        const int32_t vInt = static_cast< int32_t >( std::floor( v ) );
        if ( feature->m_point == nullptr || ( uInt - border ) < 0 || ( vInt - border ) < 0 ||
             ( uInt + border ) >= refImage.cols || ( vInt + border ) >= refImage.rows )
            continue;
        m_featureVisibility[ cntFeature ] = true;

        const double depthNorm = ( feature->m_point->m_position - C ).norm();
        const Eigen::Vector3d point( feature->m_bearingVec * depthNorm );

        /// just for test
        const double depth = frame.world2camera( feature->m_point->m_position ).z();
        const Eigen::Vector3d newPoint( feature->m_homogenous * depth );
        ///

        Eigen::Matrix< double, 2, 6 > imageJac;
        computeImageJac( imageJac, point, fx, fy );

        float* pixelPtr   = m_refPatches.ptr< float >() + cntFeature * m_patchArea;
        uint32_t cntPixel = 0;
        // FIXME: Patch size should be set as odd
        const int32_t beginIdx = -m_halfPatchSize;
        const int32_t endIdx   = m_halfPatchSize;
        for ( int32_t y{beginIdx}; y <= endIdx; y++ )
        {
            for ( int32_t x{beginIdx}; x <= endIdx; x++, cntPixel++, pixelPtr++ )
            {
                const double rowIdx = v + y;
                const double colIdx = u + x;
                *pixelPtr           = algorithm::bilinearInterpolation( refImageEigen, colIdx, rowIdx );
                const double dx     = 0.5 * ( algorithm::bilinearInterpolation( refImageEigen, colIdx + 1, rowIdx ) -
                                          algorithm::bilinearInterpolation( refImageEigen, colIdx - 1, rowIdx ) );
                const double dy     = 0.5 * ( algorithm::bilinearInterpolation( refImageEigen, colIdx, rowIdx + 1 ) -
                                          algorithm::bilinearInterpolation( refImageEigen, colIdx, rowIdx - 1 ) );
                m_jacobian.row( cntFeature * m_patchArea + cntPixel ) = dx * imageJac.row( 0 ) + dy * imageJac.row( 1 );
            }
        }
        cntFeature++;
    }
    cv::Mat visPatches;
    cv::normalize( m_refPatches, visPatches, 0, 255, cv::NORM_MINMAX, CV_32F );
    cv::imshow("refPatches", visPatches);
}

void ImageAlignment::computeResiduals( Frame& refFrame, Frame& curFrame, uint32_t level, Sophus::SE3d& pose )
{
    const cv::Mat& curImage = curFrame.m_imagePyramid.getImageAtLevel( level );
    const algorithm::MapXRowConst curImageEigen( curImage.ptr< float >(), curImage.rows, curImage.cols );
    const int32_t border    = m_halfPatchSize + 2;
    const uint32_t stride       = curImage.cols;
    const double levelDominator = 1 << level;
    const double scale          = 1.0 / levelDominator;
    const Eigen::Vector3d C     = refFrame.cameraInWorld();
    uint32_t cntFeature         = 0;
    for ( const auto& feature : refFrame.m_frameFeatures )
    {
        if (m_featureVisibility[cntFeature] == false)
            continue;

        const double depthNorm = ( feature->m_point->m_position - C ).norm();
        const Eigen::Vector3d refPoint( feature->m_bearingVec * depthNorm );
        const Eigen::Vector3d curPoint( pose * refPoint );
        const Eigen::Vector2d curFeature (curFrame.camera2image(curPoint));
        const double u     = curFeature.x() * scale;
        const double v     = curFeature.y() * scale;
        const int32_t uInt = static_cast< int32_t >( std::floor( u ) );
        const int32_t vInt = static_cast< int32_t >( std::floor( v ) );
        if ( feature->m_point == nullptr || ( uInt - border ) < 0 || ( vInt - border ) < 0 ||
             ( uInt + border ) >= curImage.cols || ( vInt + border ) >= curImage.rows )
            continue;
        
        float* pixelPtr   = m_refPatches.ptr< float >() + cntFeature * m_patchArea;
        uint32_t cntPixel = 0;
        // FIXME: Patch size should be set as odd
        const int32_t beginIdx = -m_halfPatchSize;
        const int32_t endIdx   = m_halfPatchSize;
        for ( int32_t y{beginIdx}; y <= endIdx; y++ )
        {
            for ( int32_t x{beginIdx}; x <= endIdx; x++, cntPixel++, pixelPtr++ )
            {
                const double rowIdx = v + y;
                const double colIdx = u + x;
                const float curPixelValue           = algorithm::bilinearInterpolation( curImageEigen, colIdx, rowIdx );
                m_residual(cntFeature * m_patchArea + cntPixel) = static_cast<double>(curPixelValue - *pixelPtr);
            }
        }
        cntFeature++;
    }
}

void ImageAlignment::computeImageJac( Eigen::Matrix< double, 2, 6 >& imageJac,
                                      const Eigen::Vector3d& point,
                                      const double fx,
                                      const double fy )
{
    // Image Gradient-based Joint Direct Visual Odometry for Stereo Camera, Eq. 12
    // Taking a Deeper Look at the Inverse Compositional Algorithm, Eq. 28
    // https://github.com/uzh-rpg/rpg_svo/blob/master/svo/include/svo/frame.h, jacobian_xyz2uv function but the negative
    // one

    //                              ⎡fx        -fx⋅x ⎤
    //                              ⎢──   0.0  ──────⎥
    //                              ⎢z           z₂  ⎥
    // dx / dX =                    ⎢                ⎥
    //                              ⎢     fy   -fy⋅y ⎥
    //                              ⎢0.0  ──   ──────⎥
    //                              ⎣     z      z₂  ⎦

    //                              ⎡1  0  0  0   z   -y⎤
    //                              ⎢                   ⎥
    // dX / d(theta) = [I [X]x]     ⎢0  1  0  -z  0   x ⎥
    //                              ⎢                   ⎥
    //                              ⎣0  0  1  y   -x  0 ⎦

    //                              ⎡                                  .             ⎤
    //                              ⎢fx      -fx⋅x     -fx⋅x⋅y     fx⋅x₂       -fx⋅y ⎥
    //                              ⎢──  0   ──────    ────────    ───── + fx  ──────⎥
    //                              ⎢z         z₂         z₂         z₂          z   ⎥
    // (dx / dX) * (dX / d(theta))  ⎢                                                ⎥
    //                              ⎢                      .                         ⎥
    //                              ⎢    fy  -fy⋅y     fy⋅y₂         fy⋅x⋅y     fy⋅x ⎥
    //                              ⎢0   ──  ──────  - ───── - fy    ──────     ──── ⎥
    //                              ⎣    z     z₂        z₂            z₂        z   ⎦

    const double x  = point.x();
    const double y  = point.y();
    const double z  = point.z();
    const double x2 = x * x;
    const double y2 = y * y;
    const double z2 = z * z;

    imageJac( 0, 0 ) = fx / z;
    imageJac( 0, 1 ) = 0.0;
    imageJac( 0, 2 ) = -( fx * x ) / z2;
    imageJac( 0, 3 ) = -( fx * x * y ) / z2;
    imageJac( 0, 4 ) = ( fx * x2 ) / z2 + fx;
    imageJac( 0, 5 ) = -( fx * y ) / z;

    imageJac( 1, 0 ) = 0.0;
    imageJac( 1, 1 ) = fy / z;
    imageJac( 1, 2 ) = -( fy * y ) / z2;
    imageJac( 1, 3 ) = -( fy * y2 ) / z2 - fy;
    imageJac( 1, 4 ) = ( fy * x * y ) / z2;
    imageJac( 1, 5 ) = ( fy * x ) / z;
}