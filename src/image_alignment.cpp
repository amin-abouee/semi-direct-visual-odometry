#include "image_alignment.hpp"
#include "feature.hpp"
#include "utils.hpp"
#include "visualization.hpp"

#include <easylogging++.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <sophus/se3.hpp>

#include <algorithm>

#define Alignment_Log( LEVEL ) CLOG( LEVEL, "Alignment" )

ImageAlignment::ImageAlignment( uint32_t patchSize, int32_t minLevel, int32_t maxLevel, uint32_t numParameters )
    : m_patchSize( patchSize )
    , m_halfPatchSize( patchSize / 2 )
    , m_patchArea( patchSize * patchSize )
    , m_minLevel( minLevel )
    , m_maxLevel( maxLevel )
    , m_optimizer( numParameters )
{
}

double ImageAlignment::align( std::shared_ptr< Frame >& refFrame, std::shared_ptr< Frame >& curFrame )
{
    if ( refFrame->numberObservation() == 0 )
        return 0;

    const auto& lastKF             = refFrame->m_lastKeyframe;
    const std::size_t numFeatures  = refFrame->numberObservation() + lastKF->numberObservation();

    Alignment_Log (DEBUG) << "id ref: " << refFrame->m_id << ", size observation: " << refFrame->numberObservation();
    Alignment_Log (DEBUG) << "id laskKF: " << lastKF->m_id << ", size observation: " << lastKF->numberObservation();
    const uint32_t numObservations = numFeatures * m_patchArea;
    m_refPatches.conservativeResize( numFeatures, m_patchArea );
    // m_refPatches                   = cv::Mat( numFeatures, m_patchArea, CV_32F );
    m_optimizer.initParameters( numObservations );
    m_refVisibility.resize( numFeatures, false );

    Alignment_Log (DEBUG) << "numFeatures: " << numFeatures << ", numObservation: " << numObservations;
    // Alignment_Log (DEBUG) << "patch size: " << m_refPatches.format( utils::eigenFormat() );
    // Alignment_Log (DEBUG) << "optimizer m_jacobian: " << m_optimizer.m_jacobian.format( utils::eigenFormat() );
    // Alignment_Log (DEBUG) << "optimizer m_residual: " << m_optimizer.m_residuals.format( utils::eigenFormat() );
    // Alignment_Log (DEBUG) << "optimizer m_weights: " << m_optimizer.m_weights.format( utils::eigenFormat() );
    // Alignment_Log (DEBUG) << "optimizer m_hessian: " << m_optimizer.m_hessian.format( utils::eigenFormat() );
    // Alignment_Log (DEBUG) << "optimizer m_gradient: " << m_optimizer.m_gradient.format( utils::eigenFormat() );
    // Alignment_Log (DEBUG) << "optimizer m_dx: " << m_optimizer.m_dx.format( utils::eigenFormat() );

    auto lambdaUpdateFunctor = [ this ]( Sophus::SE3d& pose, const Eigen::VectorXd& dx ) -> void { update( pose, dx ); };
    double error             = 0.0;
    Optimizer::Status optimizationStatus;

    // auto t3 = std::chrono::high_resolution_clock::now();
    // when we wanna compare a uint32 with an int32, the c++ can not compare -1 with 0
    for ( int32_t level( m_maxLevel ); level >= m_minLevel; level-- )
    {
        Alignment_Log (DEBUG) << "level: " << level;

        computeJacobian( refFrame, level );

        auto lambdaResidualFunctor = [ this, &refFrame, &curFrame, &level ]( Sophus::SE3d& pose ) -> uint32_t {
            return computeResiduals( refFrame, curFrame, level, pose );
        };
        std::tie( optimizationStatus, error ) =
          m_optimizer.optimizeLM< Sophus::SE3d >( curFrame->m_absPose, lambdaResidualFunctor, nullptr, lambdaUpdateFunctor );

        Alignment_Log (DEBUG) << "level: " << level << ", error: " << error << ", status: " << int32_t(optimizationStatus);
    }

    Alignment_Log( DEBUG ) << "Computed Pose: " << curFrame->m_absPose.params().transpose();

    // int32_t featureCounter = 0;
    // for ( uint32_t i( 0 ); i < numFeatures; i += 2 )
    // {
    //     if ( m_refVisibility[ i ] == true && m_refVisibility[ i + 1 ] == true )
    //     {
    //         const int32_t idx    = i / 2;
    //         const auto& refFeature = refFrame->m_features[ idx ];
    //         const auto& point = refFeature->m_point;
    //         const Eigen::Vector2d pixelPosition = curFrame->world2image(point->m_position);
    //         std::shared_ptr< Feature > feature = std::make_shared< Feature >( curFrame, pixelPosition, 0 );
    //         curFrame->addFeature( feature );
    //         // Here we add a reference in the feature to the 3D point, the other way
    //         // round is only done if this frame is selected as keyframe.
    //         feature->m_point = point;
    //     }
    // }

    return error;
}

void ImageAlignment::computeJacobian( const std::shared_ptr< Frame >& frame, const uint32_t level )
{
    resetParameters();
    const int32_t border    = m_halfPatchSize + 2;
    const cv::Mat& refImage = frame->m_imagePyramid.getImageAtLevel( level );
    const algorithm::MapXRowConst refImageEigen( refImage.ptr< uint8_t >(), refImage.rows, refImage.cols );

    const double levelDominator   = 1 << level;
    const double scale            = 1.0 / levelDominator;
    const Eigen::Vector3d refC    = frame->cameraInWorld();
    const double fx               = frame->m_camera->fx() / levelDominator;
    const double fy               = frame->m_camera->fy() / levelDominator;
    uint32_t cntFeature           = 0;

    Alignment_Log (DEBUG) << "level: " << level << ", scale: " << scale << ", C: " << refC.transpose() << ", fx: " << fx << ", fy: " << fy;
    // project all feature of reference frame
    for ( const auto& feature : frame->m_features )
    {
        if ( feature->m_point == nullptr )
        {
            cntFeature++;
            continue;
        }

        bool res = computeJacobianSingleFeature( feature, refImageEigen, border, refC, scale, fx, fy, cntFeature );
        if ( res == false )
        {
            cntFeature++;
            continue;
        }
    }
    Alignment_Log (DEBUG) << "after previous frame cntFeature: " << cntFeature;

    const std::shared_ptr< Frame >& lastKeyframe = frame->m_lastKeyframe;
    const cv::Mat& lastKFImage                   = lastKeyframe->m_imagePyramid.getImageAtLevel( level );
    const algorithm::MapXRowConst lastKFImageEigen( lastKFImage.ptr< uint8_t >(), lastKFImage.rows, lastKFImage.cols );
    const Eigen::Vector3d lastKFC = lastKeyframe->cameraInWorld();
    // project all feature of last keyframe
    for ( const auto& feature : lastKeyframe->m_features )
    {
        if ( feature->m_point == nullptr )
        {
            cntFeature++;
            continue;
        }

        bool res = computeJacobianSingleFeature( feature, lastKFImageEigen, border, lastKFC, scale, fx, fy, cntFeature );
        if ( res == false )
        {
            cntFeature++;
            continue;
        }
    }

    Alignment_Log (DEBUG) << "cntFeature: " << cntFeature;
    Alignment_Log (DEBUG) << "optimizer m_jacobian: " << m_optimizer.m_jacobian.format( utils::eigenFormat() );
    // visualization::templatePatches( m_refPatches, cntFeature, m_patchSize, 10, 10, 12 );
}

bool ImageAlignment::computeJacobianSingleFeature( const std::shared_ptr< Feature >& feature,
                                                   const algorithm::MapXRowConst& imageEigen,
                                                   const int32_t border,
                                                   const Eigen::Vector3d& cameraInWorld,
                                                   const double scale,
                                                   const double scaledFx,
                                                   const double scaledFy,
                                                   uint32_t& cntFeature )
{
    const auto& frame  = feature->m_frame;
    const double u     = feature->m_pixelPosition.x() * scale;
    const double v     = feature->m_pixelPosition.y() * scale;
    const int32_t uInt = static_cast< int32_t >( std::floor( u ) );
    const int32_t vInt = static_cast< int32_t >( std::floor( v ) );
    Alignment_Log (DEBUG) << "feature position: " << feature->m_pixelPosition.transpose();
    Alignment_Log (DEBUG) << "border: " << border;
    Alignment_Log (DEBUG) << "u: " << uInt << ", v: " << vInt;
    if ( ( uInt - border ) < 0 || ( vInt - border ) < 0 || ( uInt + border ) >= imageEigen.cols() ||
         ( vInt + border ) >= imageEigen.rows() )
    {
        return false;
    }


    m_refVisibility[ cntFeature ]   = true;
    const double depthNorm          = ( feature->m_point->m_position - cameraInWorld ).norm();
    const Eigen::Vector3d point3d_C = feature->m_bearingVec * depthNorm;
    const Eigen::Vector3d point3d_W = frame->camera2world( point3d_C );

    Alignment_Log (DEBUG) << "depth: " << depthNorm;
    Alignment_Log (DEBUG) << "bearing: " << feature->m_bearingVec.transpose();
    Alignment_Log (DEBUG) << "point3d_C: " << point3d_C.transpose();
    Alignment_Log (DEBUG) << "point3d_W: " << point3d_W.transpose();

    Eigen::Matrix< double, 2, 6 > imageJac;
    computeImageJac( imageJac, point3d_W, scaledFx, scaledFy );
    Alignment_Log (DEBUG) << "imageJac: " << imageJac.format( utils::eigenFormat() );


    uint32_t cntPixel = 0;
    // FIXME: Patch size should be set as odd
    int32_t beginIdx = -m_halfPatchSize;
    int32_t endIdx   = m_halfPatchSize;
    for ( int32_t y{ beginIdx }; y <= endIdx; y++ )
    {
        for ( int32_t x{ beginIdx }; x <= endIdx; x++, cntPixel++ )
        {
            const double rowIdx                  = v + y;
            const double colIdx                  = u + x;
            m_refPatches( cntFeature, cntPixel ) = algorithm::bilinearInterpolationDouble( imageEigen, colIdx, rowIdx );
            const double dx                      = 0.5 * ( algorithm::bilinearInterpolationDouble( imageEigen, colIdx + 1, rowIdx ) -
                                      algorithm::bilinearInterpolationDouble( imageEigen, colIdx - 1, rowIdx ) );
            const double dy                      = 0.5 * ( algorithm::bilinearInterpolationDouble( imageEigen, colIdx, rowIdx + 1 ) -
                                      algorithm::bilinearInterpolationDouble( imageEigen, colIdx, rowIdx - 1 ) );
            m_optimizer.m_jacobian.row( cntFeature * m_patchArea + cntPixel ) = dx * imageJac.row( 0 ) + dy * imageJac.row( 1 );

            Alignment_Log (DEBUG) << "idx " << cntFeature * m_patchArea + cntPixel;
            Alignment_Log (DEBUG) << "rowId: " << rowIdx << ", colIdx: " << colIdx;
            Alignment_Log (DEBUG) << "dx: " << dx << ", dy: " << dy;
            Alignment_Log (DEBUG) << "jac: " << m_optimizer.m_jacobian.row( cntFeature * m_patchArea + cntPixel ).format( utils::eigenFormat() );
        }
    }
    cntFeature++;
    return true;
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

// if we define the residual error as current image - reference image, we do not need to apply the negative for gradient
uint32_t ImageAlignment::computeResiduals( const std::shared_ptr< Frame >& refFrame,
                                           const std::shared_ptr< Frame >& curFrame,
                                           const uint32_t level,
                                           const Sophus::SE3d& pose )
{
    const std::shared_ptr< Frame >& lastKeyframe = refFrame->m_lastKeyframe;
    const cv::Mat& curImage                      = curFrame->m_imagePyramid.getImageAtLevel( level );
    const algorithm::MapXRowConst curImageEigen( curImage.ptr< uint8_t >(), curImage.rows, curImage.cols );
    const int32_t border = m_halfPatchSize + 2;
    // const uint32_t stride            = curImage.cols;
    const double levelDominator      = 1 << level;
    const double scale               = 1.0 / levelDominator;
    const Eigen::Vector3d refC       = refFrame->cameraInWorld();
    uint32_t cntFeature              = 0;
    uint32_t cntTotalProjectedPixels = 0;

    // Alignment_Log (DEBUG) << "border: " << border;
    // Alignment_Log (DEBUG) << "refC: " << refC.transpose();
    // Alignment_Log (DEBUG) << "scale: " << scale;
    
    for ( const auto& feature : refFrame->m_features )
    {
        if ( m_refVisibility[ cntFeature ] == false )
        {
            cntFeature ++;
            continue;
        }
        bool res =
          computeResidualSingleFeature( feature, curImageEigen, curFrame, pose, border, refC, scale, cntFeature, cntTotalProjectedPixels );
        if ( res == false )
        {
            // we also ignore the feature of last keyframe
            cntFeature ++;
            continue;
        }
    }

    const Eigen::Vector3d lastKFC    = lastKeyframe->cameraInWorld();
    for (const auto& feature : lastKeyframe->m_features)
    {   
        if ( m_refVisibility[ cntFeature ] == false )
        {
            cntFeature ++;
            continue;
        }
        bool res = computeResidualSingleFeature( feature, curImageEigen, curFrame, pose, border, lastKFC, scale, cntFeature,
                                            cntTotalProjectedPixels );
        if ( res == false )
        {
            cntFeature++;
            continue;
        }
    }
    Alignment_Log (DEBUG) << "cntFeature: " << cntFeature;
    Alignment_Log (DEBUG) << "cntTotalProjectedPixels: " << cntTotalProjectedPixels;

    return cntTotalProjectedPixels;
}

bool ImageAlignment::computeResidualSingleFeature( const std::shared_ptr< Feature >& feature,
                                                   const algorithm::MapXRowConst& imageEigen,
                                                   const std::shared_ptr< Frame >& curFrame,
                                                   const Sophus::SE3d& pose,
                                                   const int32_t border,
                                                   const Eigen::Vector3d& cameraInWorld,
                                                   const double scale,
                                                   uint32_t& cntFeature,
                                                   uint32_t& cntTotalProjectedPixels )
{
    const auto& refFrame             = feature->m_frame;
    const double depthNorm           = ( feature->m_point->m_position - cameraInWorld ).norm();
    const Eigen::Vector3d point3d_C  = feature->m_bearingVec * depthNorm;
    const Eigen::Vector3d point3d_W  = refFrame->camera2world( point3d_C );
    const Eigen::Vector3d curPoint   = pose * point3d_W;
    const Eigen::Vector2d curFeature = curFrame->camera2image( curPoint );
    const double u                   = curFeature.x() * scale;
    const double v                   = curFeature.y() * scale;
    const int32_t uInt               = static_cast< int32_t >( std::floor( u ) );
    const int32_t vInt               = static_cast< int32_t >( std::floor( v ) );

    Alignment_Log (DEBUG) << "feature position: " << feature->m_pixelPosition.transpose();
    Alignment_Log (DEBUG) << "depthNorm: " << depthNorm;
    Alignment_Log (DEBUG) << "curFeature " << curFeature.transpose();
    Alignment_Log (DEBUG) << "u: " << uInt << ", v: " << vInt;

    if ( feature->m_point == nullptr || ( uInt - border ) < 0 || ( vInt - border ) < 0 || ( uInt + border ) >= imageEigen.cols() ||
         ( vInt + border ) >= imageEigen.rows() )
    {
        return false;
    }

    uint32_t cntPixel = 0;
    // FIXME: Patch size should be set as odd
    const int32_t beginIdx = -m_halfPatchSize;
    const int32_t endIdx   = m_halfPatchSize;
    for ( int32_t y{ beginIdx }; y <= endIdx; y++ )
    {
        for ( int32_t x{ beginIdx }; x <= endIdx; x++, cntPixel++, cntTotalProjectedPixels++ )
        {
            const double rowIdx        = v + y;
            const double colIdx        = u + x;
            const double curPixelValue = algorithm::bilinearInterpolationDouble( imageEigen, colIdx, rowIdx );

            // ****
            // IF we compute the error of inverse compositional as r = T(x) - I(W), then we should solve (delta p) = -(JtWT).inverse() *
            // JtWr BUt if we take r = I(W) - T(x) a residual error, then (delta p) = (JtWT).inverse() * JtWr
            // ***

            m_optimizer.m_residuals( cntFeature * m_patchArea + cntPixel )     = curPixelValue - m_refPatches( cntFeature, cntPixel );
            m_optimizer.m_visiblePoints( cntFeature * m_patchArea + cntPixel ) = true;

            Alignment_Log (DEBUG) << "idx " << cntFeature * m_patchArea + cntPixel;
            Alignment_Log (DEBUG) << "rowId: " << rowIdx << ", colIdx: " << colIdx;
            Alignment_Log (DEBUG) << "curPixelValue: " << curPixelValue;
            Alignment_Log (DEBUG) << "res " << m_optimizer.m_residuals( cntFeature * m_patchArea + cntPixel );
        }
    }
    cntFeature++;
    return true;
}

void ImageAlignment::update( Sophus::SE3d& pose, const Eigen::VectorXd& dx )
{
    // pose = Sophus::SE3d::exp( dx ) * pose;
    // std::cout << "Update with inverse: " << Sophus::SE3d::exp( dx ).inverse().params().transpose() << std::endl;
    // std::cout << "Update with minus: " << Sophus::SE3d::exp( -dx ).params().transpose() << std::endl;
    // inverse a SE3d pose is equivalent with getting the pose from minus of lie algebra parameters.
    // Sophus::SE3d::exp( dx ).inverse() == Sophus::SE3d::exp( -dx )
    pose = pose * Sophus::SE3d::exp( -dx );
}

void ImageAlignment::resetParameters()
{
    std::fill( m_refVisibility.begin(), m_refVisibility.end(), false );
    // m_refPatches = cv::Scalar( 0.0 );
    m_refPatches.setZero();
}
