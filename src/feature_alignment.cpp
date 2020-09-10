#include "feature_alignment.hpp"
#include "algorithm.hpp"
#include "feature.hpp"
#include "utils.hpp"
#include "visualization.hpp"

#include <algorithm>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "easylogging++.h"
#define Alignment_Log( LEVEL ) CLOG( LEVEL, "Alignment" )

FeatureAlignment::FeatureAlignment( uint32_t patchSize, int32_t level, uint32_t numParameters )
    : m_patchSize( patchSize )
    , m_halfPatchSize( patchSize / 2 )
    , m_patchArea( patchSize * patchSize )
    , m_level( level )
    , m_optimizer( numParameters )
{
    // el::Loggers::getLogger( "Tracker" );  // Register new logger
    // std::cout << "c'tor image alignment" << std::endl;
}

double FeatureAlignment::align( const std::shared_ptr< Feature >& refFeature,
                                const std::shared_ptr< Frame >& curFrame,
                                Eigen::Vector2d& pixelPos )
{
    const std::size_t numFeatures  = 1;
    const uint32_t numObservations = numFeatures * m_patchArea;
    m_refPatches                   = cv::Mat( 1, m_patchArea, CV_32F );
    m_optimizer.initParameters( numObservations );
    m_refVisibility.resize( numFeatures, false );

    auto lambdaUpdateFunctor = [ this ]( Sophus::SE2d& pose, const Eigen::Vector3d& dx ) -> void { update( pose, dx ); };
    double error             = 0.0;
    Optimizer::Status optimizationStatus;
    Sophus::SE2d relativePose;

    // t1 = std::chrono::high_resolution_clock::now();
    computeJacobian( refFeature, 0 );
    // timerJacobian += std::chrono::duration_cast< std::chrono::microseconds >( std::chrono::high_resolution_clock::now() - t1
    // ).count();

    auto lambdaResidualFunctor = [ this, &refFeature, &curFrame ]( Sophus::SE2d& pose ) -> uint32_t {
        return computeResiduals( refFeature, curFrame, 0, pose );
    };
    // t1 = std::chrono::high_resolution_clock::now();
    std::tie( optimizationStatus, error ) = m_optimizer.optimizeGN<Sophus::SE2d>( relativePose, lambdaResidualFunctor, nullptr, lambdaUpdateFunctor );
    pixelPos = relativePose * refFeature->m_feature;
    return error;
}

void FeatureAlignment::computeJacobian( const std::shared_ptr< Feature >& refFeature, const uint32_t level )
{
    resetParameters();
    const int32_t border = m_halfPatchSize + 2;
    const cv::Mat& refImage = refFeature->m_frame->m_imagePyramid.getImageAtLevel( level );
    const algorithm::MapXRowConst refImageEigen( refImage.ptr< uint8_t >(), refImage.rows, refImage.cols );
    const Eigen::Vector2d pixelPos = refFeature->m_feature;

    if ( refFeature->m_point == nullptr || refFeature->m_frame->m_camera->isInFrame(pixelPos, border))
    {
        return;
    }

    float* pixelPtr   = m_refPatches.row( 0 ).ptr< float >();
    uint32_t cntPixel = 0;
    // FIXME: Patch size should be set as odd
    const int32_t beginIdx = -m_halfPatchSize;
    const int32_t endIdx   = m_halfPatchSize;
    for ( int32_t y{ beginIdx }; y <= endIdx; y++ )
    {
        for ( int32_t x{ beginIdx }; x <= endIdx; x++, cntPixel++, pixelPtr++ )
        {
            Eigen::Matrix< double, 2, 3 > imageJac;
            computeImageJac( imageJac, pixelPos );

            const double rowIdx                    = pixelPos.y() + y;
            const double colIdx                    = pixelPos.x() + x;
            *pixelPtr                              = algorithm::bilinearInterpolation( refImageEigen, colIdx, rowIdx );
            const double dx                        = 0.5 * ( algorithm::bilinearInterpolation( refImageEigen, colIdx + 1, rowIdx ) -
                                      algorithm::bilinearInterpolation( refImageEigen, colIdx - 1, rowIdx ) );
            const double dy                        = 0.5 * ( algorithm::bilinearInterpolation( refImageEigen, colIdx, rowIdx + 1 ) -
                                      algorithm::bilinearInterpolation( refImageEigen, colIdx, rowIdx - 1 ) );
            m_optimizer.m_jacobian.row( cntPixel ) = dx * imageJac.row( 0 ) + dy * imageJac.row( 1 );
        }
    }
    m_refVisibility[ 0 ] = false;
    // cntFeature++;
    // visualization::templatePatches( m_refPatches, cntFeature, m_patchSize, 10, 10, 12 );
}

// if we define the residual error as current image - reference image, we do not need to apply the negative for gradient
uint32_t FeatureAlignment::computeResiduals( const std::shared_ptr< Feature >& refFeature,
                                             const std::shared_ptr< Frame >& curFrame,
                                             const uint32_t level,
                                             Sophus::SE2d& pose )
{
    const int32_t border = m_halfPatchSize + 2;
    const cv::Mat& curImage = curFrame->m_imagePyramid.getImageAtLevel( level );
    const algorithm::MapXRowConst curImageEigen( curImage.ptr< uint8_t >(), curImage.rows, curImage.cols );
    const Eigen::Vector2d pixelPos = refFeature->m_feature;
    // const uint32_t stride            = curImage.cols;

    if ( m_refVisibility[ 0 ] == false )
    {
        return 0;
    }

    curFrame->m_camera->isInFrame(pixelPos, border);
    {
        return 0;
    }

    uint32_t cntTotalProjectedPixels = 0;
    float* pixelPtr   = m_refPatches.row( 0 ).ptr< float >();
    uint32_t cntPixel = 0;
    // FIXME: Patch size should be set as odd
    const int32_t beginIdx = -m_halfPatchSize;
    const int32_t endIdx   = m_halfPatchSize;
    for ( int32_t y{ beginIdx }; y <= endIdx; y++ )
    {
        for ( int32_t x{ beginIdx }; x <= endIdx; x++, cntPixel++, pixelPtr++, cntTotalProjectedPixels++ )
        {
            const Eigen::Vector2d warpedPoint = pose * Eigen::Vector2d(x, y);
            const double rowIdx       = pixelPos.y() + warpedPoint.y();
            const double colIdx       = pixelPos.x() + warpedPoint.x();
            const float curPixelValue = algorithm::bilinearInterpolation( curImageEigen, colIdx, rowIdx );

            // ****
            // IF we compute the error of inverse compositional as r = T(x) - I(W), then we should solve (delta p) = -(JtWT).inverse() *
            // JtWr BUt if we take r = I(W) - T(x) a residual error, then (delta p) = (JtWT).inverse() * JtWr
            // ***

            m_optimizer.m_residuals( cntPixel ) = static_cast< double >( curPixelValue - *pixelPtr );
            // m_optimizer.m_residuals( cntFeature * m_patchArea + cntPixel ) = static_cast< double >( *pixelPtr - curPixelValue);
            m_optimizer.m_visiblePoints( cntPixel ) = true;
        }
    }
    return cntTotalProjectedPixels;
}

void FeatureAlignment::computeImageJac( Eigen::Matrix< double, 2, 3 >& imageJac, const Eigen::Vector2d& pixelPos )
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

    const double x = pixelPos.x();
    const double y = pixelPos.y();

    imageJac( 0, 0 ) = 1.0;
    imageJac( 0, 1 ) = 0.0;
    imageJac( 0, 2 ) = -y;

    imageJac( 1, 0 ) = 0.0;
    imageJac( 1, 1 ) = 1.0;
    imageJac( 1, 2 ) = x;
}

void FeatureAlignment::update( Sophus::SE2d& pose, const Eigen::Vector3d& dx )
{
    // pose = Sophus::SE3d::exp( dx ) * pose;
    // std::cout << "Update with inverse: " << Sophus::SE3d::exp( dx ).inverse().params().transpose() << std::endl;
    // std::cout << "Update with minus: " << Sophus::SE3d::exp( -dx ).params().transpose() << std::endl;
    // inverse a SE3d pose is equivalent with getting the pose from minus of lie algebra parameters.
    // Sophus::SE3d::exp( dx ).inverse() == Sophus::SE3d::exp( -dx )
    pose = pose * Sophus::SE2d::exp( -dx );
}

void FeatureAlignment::resetParameters()
{
    std::fill( m_refVisibility.begin(), m_refVisibility.end(), false );
    m_refPatches = cv::Scalar( 0.0 );
}
