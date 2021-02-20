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

}

double FeatureAlignment::align( const std::shared_ptr< Feature >& refFeature,
                                const std::shared_ptr< Frame >& curFrame,
                                Eigen::Vector2d& pixelPos )
{
    const uint32_t numObservations = m_patchArea;
    m_refPatches.resize( numObservations );
    m_optimizer.initParameters( numObservations );
    m_refVisibility.resize( numObservations, false );

    auto lambdaUpdateFunctor = [ this ]( Eigen::Vector3d& pose, const Eigen::Vector3d& dx ) -> void { update( pose, dx ); };
    double error             = 0.0;
    Optimizer::Status optimizationStatus;
    // Sophus::SE2d relativePose;
    // relativePose.translation() = pixelPos - refFeature->m_pixelPosition;

    // Alignment_Log( DEBUG ) << "RelativePose: " << relativePose.params().transpose();

    Eigen::Vector3d flow (0.0, 0.0, 0.0);
    flow.x() = pixelPos.x();
    flow.y() = pixelPos.y();

    computeJacobian( refFeature );

    auto lambdaResidualFunctor = [ this, &refFeature, &curFrame ]( Eigen::Vector3d& pose ) -> uint32_t {
        return computeResiduals( refFeature, curFrame, pose );
    };

    std::tie( optimizationStatus, error ) =
      m_optimizer.optimizeLM< Eigen::Vector3d >( flow, lambdaResidualFunctor, nullptr, lambdaUpdateFunctor );

    Alignment_Log( DEBUG ) << "error: " << error << ", status: " << uint32_t ( optimizationStatus ) << ", flow: " << flow.transpose();

    // pixelPos = relativePose * refFeature->m_pixelPosition;
    pixelPos.x() = flow.x();
    pixelPos.y() = flow.y();

    return error;
}

void FeatureAlignment::computeJacobian( const std::shared_ptr< Feature >& refFeature )
{
    resetParameters();

    const double border     = m_halfPatchSize + 2;
    const cv::Mat& refImage = refFeature->m_frame->m_imagePyramid.getGradientAtLevel( 0 );
    const algorithm::MapXRowConst refImageEigen( refImage.ptr< uint8_t >(), refImage.rows, refImage.cols );
    const Eigen::Vector2d pixelPos = refFeature->m_pixelPosition;

    // check for unsuccessful points
    if ( refFeature->m_frame->m_camera->isInFrame( pixelPos, border ) == false )
    {
        return;
    }

    uint32_t cntPixel = 0;
    // FIXME: Patch size should be set as odd
    const int32_t beginIdx = -m_halfPatchSize;
    const int32_t endIdx   = m_halfPatchSize;
    for ( int32_t y( beginIdx ); y <= endIdx; y++ )
    {
        for ( int32_t x( beginIdx ); x <= endIdx; x++, cntPixel++ )
        {
            // Eigen::Matrix< double, 2, 3 > imageJac;
            // computeImageJac( imageJac, pixelPos );

            const double rowIdx = pixelPos.y() + y;
            const double colIdx = pixelPos.x() + x;
            // *pixelPtr                              = algorithm::bilinearInterpolation( refImageEigen, colIdx, rowIdx );
            m_refPatches( cntPixel )               = algorithm::bilinearInterpolation( refImageEigen, colIdx, rowIdx );
            const double dx                        = 0.5 * ( algorithm::bilinearInterpolation( refImageEigen, colIdx + 1, rowIdx ) -
                                      algorithm::bilinearInterpolation( refImageEigen, colIdx - 1, rowIdx ) );
            const double dy                        = 0.5 * ( algorithm::bilinearInterpolation( refImageEigen, colIdx, rowIdx + 1 ) -
                                      algorithm::bilinearInterpolation( refImageEigen, colIdx, rowIdx - 1 ) );
            // m_optimizer.m_jacobian.row( cntPixel ) = dx * imageJac.row( 0 ) + dy * imageJac.row( 1 );
            m_optimizer.m_jacobian.row( cntPixel ) = Eigen::Vector3d(dx, dy, 1.0);
        }
    }
    // m_refVisibility[ 0 ] = true;
    // cntFeature++;
    // cv::Mat visPatch = visualization::referencePatches( m_refPatches, 1, m_halfPatchSize, 10, 10, 12 );
    // cv::imwrite( "../output/images/refPatch.png", visPatch );

    // cv::Mat refPath = refImage( cv::Rect( pixelPos.x() - 30, pixelPos.y() - 30, 60, 60 ) );
    // cv::imwrite( "../output/images/refPatch.png", refPath );
    // cv:imshow("ref_patch", refPath);
}

// if we define the residual error as current image - reference image, we do not need to apply the negative for gradient
uint32_t FeatureAlignment::computeResiduals( const std::shared_ptr< Feature >& refFeature,
                                             const std::shared_ptr< Frame >& curFrame,
                                             Eigen::Vector3d& pose )
{
    const double border     = m_halfPatchSize + 2;
    const cv::Mat& curImage = curFrame->m_imagePyramid.getGradientAtLevel( 0 );
    const algorithm::MapXRowConst curImageEigen( curImage.ptr< uint8_t >(), curImage.rows, curImage.cols );
    const Eigen::Vector2d pixelPos = refFeature->m_pixelPosition;
    // Eigen::Vector2d warpedPoint;
    // warpedPoint.x() = pixelPos.x() + pose.x();
    // warpedPoint.y() = pixelPos.y() + pose.y();
    // const uint32_t stride            = curImage.cols;

    if ( curFrame->m_camera->isInFrame( pose.head(2), border ) == false )
    {
        return 0;
    }

    uint32_t cntTotalProjectedPixels = 0;
    // float* pixelPtr                  = m_refPatches.row( 0 ).ptr< float >();
    uint32_t cntPixel = 0;
    // FIXME: Patch size should be set as odd
    const int32_t beginIdx = -m_halfPatchSize;
    const int32_t endIdx   = m_halfPatchSize;
    for ( int32_t y{ beginIdx }; y <= endIdx; y++ )
    {
        for ( int32_t x{ beginIdx }; x <= endIdx; x++, cntPixel++, cntTotalProjectedPixels++ )
        {
            // we apply or warping to all pixels of out patch
            // warpedPoint = pose * Eigen::Vector2d( x, y );
            const double rowIdx        = pose.y() + y;
            const double colIdx        = pose.x() + x;
            const double curPixelValue = algorithm::bilinearInterpolation( curImageEigen, colIdx, rowIdx );

            // ****
            // IF we compute the error of inverse compositional as r = T(x) - I(W), then we should solve (delta p) = -(JtWT).inverse() *
            // JtWr BUt if we take r = I(W) - T(x) a residual error, then (delta p) = (JtWT).inverse() * JtWr
            // ***

            m_optimizer.m_residuals( cntPixel ) =  - (curPixelValue - m_refPatches( cntPixel ) + pose.z());
            // m_optimizer.m_residuals( cntFeature * m_patchArea + cntPixel ) = static_cast< double >( *pixelPtr - curPixelValue);
            m_optimizer.m_visiblePoints( cntPixel ) = true;
        }
    }
    // std::cout << "Res: " << m_optimizer.m_residuals.transpose() << std::endl;

    // const Eigen::Vector2d warpedPoint = pose * pixelPos;
    // cv::Mat curPath = curImage( cv::Rect( warpedPoint.x() - 30, warpedPoint.y() - 30, 60, 60 ) );
    // cv::imwrite( "../output/images/curPatch.png", curPath );
    // cv:imshow("cur_patch", curPath);
    // cv::waitKey(0);

    return cntTotalProjectedPixels;
    // cv::Mat visPatch = visualization::residualsPatches( m_optimizer.m_residuals, 1, m_halfPatchSize, 10, 10, 12 );
    // cv::imwrite( "../output/images/resPatch.png", visPatch );
}

// void FeatureAlignment::computeImageJac( Eigen::Matrix< double, 2, 3 >& imageJac, const Eigen::Vector2d& pixelPos )
// {
//     // Image Gradient-based Joint Direct Visual Odometry for Stereo Camera, Eq. 12
//     // Taking a Deeper Look at the Inverse Compositional Algorithm, Eq. 28
//     // https://github.com/uzh-rpg/rpg_svo/blob/master/svo/include/svo/frame.h, jacobian_xyz2uv function but the negative
//     // one

//     // Euclidean (tx,ty,θ)

//     //  [[cosθ −sinθ tx]
//     //   [sinθ cosθ  ty]
//     //   [0      0    1]]

//     // Jac = [[1        0     −xsinθ−ycosθ]
//     //        [0        1      xcosθ−ysinθ]]

//     // θ = 0

//     const double x = pixelPos.x();
//     const double y = pixelPos.y();

//     imageJac( 0, 0 ) = 1.0;
//     imageJac( 0, 1 ) = 0.0;
//     imageJac( 0, 2 ) = -y;

//     imageJac( 1, 0 ) = 0.0;
//     imageJac( 1, 1 ) = 1.0;
//     imageJac( 1, 2 ) = x;
// }

void FeatureAlignment::update( Eigen::Vector3d& pose, const Eigen::Vector3d& dx )
{
    pose.x() += dx.x();
    pose.y() += dx.y();
    pose.z() += dx.z();
}

void FeatureAlignment::resetParameters()
{
    std::fill( m_refVisibility.begin(), m_refVisibility.end(), false );
    // m_refPatches = cv::Scalar( 0.0 );
    m_refPatches.setZero();
}
