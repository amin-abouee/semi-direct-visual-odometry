#include "image_alignment.hpp"
#include "algorithm.hpp"
#include "feature.hpp"
#include "utils.hpp"
#include "visualization.hpp"

#include <algorithm>

#include <sophus/se3.hpp>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "easylogging++.h"
#define Alignment_Log( LEVEL ) CLOG( LEVEL, "Alignment" )

ImageAlignment::ImageAlignment( uint32_t patchSize, int32_t minLevel, int32_t maxLevel, uint32_t numParameters )
    : m_patchSize( patchSize )
    , m_halfPatchSize( patchSize / 2 )
    , m_patchArea( patchSize * patchSize )
    , m_minLevel( minLevel )
    , m_maxLevel( maxLevel )
    , m_optimizer( numParameters )
{
    // el::Loggers::getLogger( "Tracker" );  // Register new logger
    // std::cout << "c'tor image alignment" << std::endl;
}

double ImageAlignment::align( std::shared_ptr< Frame >& refFrame, std::shared_ptr< Frame >& curFrame )
{
    if ( refFrame->numberObservation() == 0 )
        return 0;

    // auto t1 = std::chrono::high_resolution_clock::now();
    const std::size_t numFeatures  = refFrame->numberObservation();
    const uint32_t numObservations = numFeatures * m_patchArea;
    m_refPatches                   = cv::Mat( numFeatures, m_patchArea, CV_32F );
    m_optimizer.initParameters( numObservations );
    m_refVisibility.resize( numFeatures, false );
    // std::cout << "Init: " << std::chrono::duration_cast< std::chrono::microseconds >( std::chrono::high_resolution_clock::now() - t1
    // ).count() << std::endl;

    // m_curVisibility.resize( numFeatures, false );

    // std::cout << "jacobian size: " << m_optimizer.m_jacobian.rows() << " , " << m_optimizer.m_jacobian.cols() << std::endl;
    // std::cout << "residuals size: " << m_optimizer.m_residuals.rows() << " , " << m_optimizer.m_residuals.cols() << std::endl;
    // std::cout << "reference Patch size: " << m_refPatches.size << std::endl;
    // std::cout << "number Observation: " << numObservations << std::endl;

    Sophus::SE3d relativePose;

    // auto lambdaJacobianFunctor = [&refFrame, level](
    //                                 Sophus::SE3d& pose ) -> void {
    //     return computeJacobian( refFrame, level );
    // };

    auto lambdaUpdateFunctor = [this]( Sophus::SE3d& pose, const Eigen::VectorXd& dx ) -> void { update( pose, dx ); };
    double error             = 0.0;
    Optimizer::Status optimizationStatus;

    // uint64_t timerJacobian = 0;
    // uint64_t timeroptimize = 0;

    // auto t3 = std::chrono::high_resolution_clock::now();
    // when we wanna compare a uint32 with an int32, the c++ can not compare -1 with 0
    for ( int32_t level( m_maxLevel ); level >= m_minLevel; level-- )
    {
        // t1 = std::chrono::high_resolution_clock::now();
        computeJacobian( refFrame, level );
        // timerJacobian += std::chrono::duration_cast< std::chrono::microseconds >( std::chrono::high_resolution_clock::now() - t1
        // ).count();

        auto lambdaResidualFunctor = [this, &refFrame, &curFrame, &level]( Sophus::SE3d& pose ) -> uint32_t {
            return computeResiduals( refFrame, curFrame, level, pose );
        };
        // t1 = std::chrono::high_resolution_clock::now();
        std::tie( optimizationStatus, error ) = m_optimizer.optimizeLM( relativePose, lambdaResidualFunctor, nullptr, lambdaUpdateFunctor );
        // timeroptimize += std::chrono::duration_cast< std::chrono::microseconds >( std::chrono::high_resolution_clock::now() - t1
        // ).count();

        // std::cout << "error at level " << level << " is: "<< error << " with status: " <<
        // static_cast<std::underlying_type<NLLS::Status>::type>(optimizationStatus) << std::endl; std::cout << "error at level " << level
        // << " is: "<< error << " with status: " << static_cast<uint32_t>(optimizationStatus) << std::endl;
    }
    // std::cout << "total: " << std::chrono::duration_cast< std::chrono::microseconds >( std::chrono::high_resolution_clock::now() - t3
    // ).count() << std::endl; std::cout << "timerJacobian: " << timerJacobian << std::endl; std::cout << "timeroptimize: " << timeroptimize
    // << std::endl; std::cout << "m_timerResiduals: " << m_optimizer.m_timerResiduals << std::endl; std::cout << "m_timerSolve: " <<
    // m_optimizer.m_timerSolve << std::endl; std::cout << "m_timerHessina: " << m_optimizer.m_timerHessian << std::endl; std::cout <<
    // "m_timerLambda: " << m_optimizer.m_timerLambda << std::endl; std::cout << "m_timerSwitch: " << m_optimizer.m_timerSwitch <<
    // std::endl; std::cout << "m_timerLambda: " << m_optimizer.m_timerLambda << std::endl; std::cout << "m_timerUpdateParameters: " <<
    // m_optimizer.m_timerUpdateParameters << std::endl; std::cout << "m_timerCheck: " << m_optimizer.m_timerCheck << std::endl; std::cout
    // << "m_timerFor: " << m_optimizer.m_timerFor << std::endl;

    // curFrame->m_TransW2F = refFrame->m_TransW2F * relativePose;
    curFrame->m_TransW2F = relativePose * refFrame->m_TransW2F;
    Alignment_Log( DEBUG ) << "Computed Pose: " << curFrame->m_TransW2F.params().transpose();
    return error;
}

void ImageAlignment::computeJacobian( std::shared_ptr< Frame >& frame, uint32_t level )
{
    resetParameters();
    const int32_t border    = m_halfPatchSize + 2;
    const cv::Mat& refImage = frame->m_imagePyramid.getImageAtLevel( level );
    // std::cout << "Type: " << refImage.type() << std::endl;
    // std::cout << "Image data:\n" << refImage << std::endl;
    const algorithm::MapXRowConst refImageEigen( refImage.ptr< uint8_t >(), refImage.rows, refImage.cols );
    // std::cout << "Image data:\n" << refImageEigen.block(0, 0, 20, 20) << std::endl;
    // const uint32_t stride       = refImage.cols;
    const double levelDominator = 1 << level;
    const double scale          = 1.0 / levelDominator;
    const Eigen::Vector3d C     = frame->cameraInWorld();
    const double fx             = frame->m_camera->fx() / levelDominator;
    const double fy             = frame->m_camera->fy() / levelDominator;
    uint32_t cntFeature         = 0;
    for ( const auto& feature : frame->m_frameFeatures )
    {
        const double u     = feature->m_feature.x() * scale;
        const double v     = feature->m_feature.y() * scale;
        const int32_t uInt = static_cast< int32_t >( std::floor( u ) );
        const int32_t vInt = static_cast< int32_t >( std::floor( v ) );
        // std::cout << "index feature: " << cntFeature << ", loc [" << uInt << " , " << vInt << "]" << std::endl;
        if ( feature->m_point == nullptr || ( uInt - border ) < 0 || ( vInt - border ) < 0 || ( uInt + border ) >= refImage.cols ||
             ( vInt + border ) >= refImage.rows )
        {
            cntFeature++;
            continue;
        }
        m_refVisibility[ cntFeature ] = true;

        const double depthNorm = ( feature->m_point->m_position - C ).norm();
        const Eigen::Vector3d point( feature->m_bearingVec * depthNorm );

        /// just for test
        const double depth = frame->world2camera( feature->m_point->m_position ).z();
        const Eigen::Vector3d newPoint( feature->m_homogenous * depth );
        ///

        Eigen::Matrix< double, 2, 6 > imageJac;
        computeImageJac( imageJac, point, fx, fy );

        float* pixelPtr   = m_refPatches.row( cntFeature ).ptr< float >();
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
                m_optimizer.m_jacobian.row( cntFeature * m_patchArea + cntPixel ) = dx * imageJac.row( 0 ) + dy * imageJac.row( 1 );
                // std::cout << "index: " << cntFeature * m_patchArea + cntPixel << std::endl;
                // std::cout << "dx: " << dx << "   row 0: " << imageJac.row( 0 ) << std::endl;
                // std::cout << "dy: " << dy << "   row 1: " << imageJac.row( 1 ) << std::endl;
                // std::cout << "jac " << m_optimizer.m_jacobian.row( cntFeature * m_patchArea + cntPixel ) <<
                // std::endl;
            }
        }
        cntFeature++;
    }
    // visualization::templatePatches( m_refPatches, cntFeature, m_patchSize, 10, 10, 12 );
}

// if we define the residual error as current image - reference image, we do not need to apply the negative for gradient
uint32_t ImageAlignment::computeResiduals( std::shared_ptr< Frame >& refFrame,
                                           std::shared_ptr< Frame >& curFrame,
                                           uint32_t level,
                                           Sophus::SE3d& pose )
{
    const cv::Mat& curImage = curFrame->m_imagePyramid.getImageAtLevel( level );
    const algorithm::MapXRowConst curImageEigen( curImage.ptr< uint8_t >(), curImage.rows, curImage.cols );
    const int32_t border = m_halfPatchSize + 2;
    // const uint32_t stride            = curImage.cols;
    const double levelDominator      = 1 << level;
    const double scale               = 1.0 / levelDominator;
    const Eigen::Vector3d C          = refFrame->cameraInWorld();
    uint32_t cntFeature              = 0;
    uint32_t cntTotalProjectedPixels = 0;
    for ( const auto& feature : refFrame->m_frameFeatures )
    {
        if ( m_refVisibility[ cntFeature ] == false )
        {
            cntFeature++;
            continue;
        }

        const double depthNorm = ( feature->m_point->m_position - C ).norm();
        const Eigen::Vector3d refPoint( feature->m_bearingVec * depthNorm );
        const Eigen::Vector3d curPoint( pose * refPoint );
        const Eigen::Vector2d curFeature( curFrame->camera2image( curPoint ) );
        const double u     = curFeature.x() * scale;
        const double v     = curFeature.y() * scale;
        const int32_t uInt = static_cast< int32_t >( std::floor( u ) );
        const int32_t vInt = static_cast< int32_t >( std::floor( v ) );
        // std::cout << "index feature: " << cntFeature << ", loc [" << uInt << " , " << vInt << "]" << std::endl;
        if ( feature->m_point == nullptr || ( uInt - border ) < 0 || ( vInt - border ) < 0 || ( uInt + border ) >= curImage.cols ||
             ( vInt + border ) >= curImage.rows )
        {
            cntFeature++;
            continue;
        }
        // m_curVisibility[ cntFeature ] = true;

        float* pixelPtr   = m_refPatches.row( cntFeature ).ptr< float >();
        uint32_t cntPixel = 0;
        // FIXME: Patch size should be set as odd
        const int32_t beginIdx = -m_halfPatchSize;
        const int32_t endIdx   = m_halfPatchSize;
        for ( int32_t y{beginIdx}; y <= endIdx; y++ )
        {
            for ( int32_t x{beginIdx}; x <= endIdx; x++, cntPixel++, pixelPtr++, cntTotalProjectedPixels++ )
            {
                const double rowIdx       = v + y;
                const double colIdx       = u + x;
                const float curPixelValue = algorithm::bilinearInterpolation( curImageEigen, colIdx, rowIdx );

                // ****
                // IF we compute the error of inverse compositional as r = T(x) - I(W), then we should solve (delta p) = -(JtWT).inverse() *
                // JtWr BUt if we take r = I(W) - T(x) a residual error, then (delta p) = (JtWT).inverse() * JtWr
                // ***

                m_optimizer.m_residuals( cntFeature * m_patchArea + cntPixel ) = static_cast< double >( curPixelValue - *pixelPtr );
                // m_optimizer.m_residuals( cntFeature * m_patchArea + cntPixel ) = static_cast< double >( *pixelPtr - curPixelValue);
                m_optimizer.m_visiblePoints( cntFeature * m_patchArea + cntPixel ) = true;
            }
        }
        cntFeature++;
    }
    return cntTotalProjectedPixels;
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
    m_refPatches = cv::Scalar( 0.0 );
}
