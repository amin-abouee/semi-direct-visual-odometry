#include "bundle_adjustment.hpp"
#include "algorithm.hpp"
#include "feature.hpp"
#include "utils.hpp"
#include "visualization.hpp"

#include <algorithm>
#include <deque>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "easylogging++.h"
#define Adjustment_Log( LEVEL ) CLOG( LEVEL, "Adjustment" )

BundleAdjustment::BundleAdjustment( int32_t level, uint32_t numParameters ) : m_level( level ), m_optimizer( numParameters )
{
    // el::Loggers::getLogger( "Tracker" );  // Register new logger
    // std::cout << "c'tor image alignment" << std::endl;
}

double BundleAdjustment::optimizePose( std::shared_ptr< Frame >& frame )
{
    if ( frame->numberObservation() == 0 )
        return 0;

    m_optimizer.setNumUnknowns(6);
    // auto t1 = std::chrono::high_resolution_clock::now();
    const std::size_t numFeatures = frame->numberObservation();
    // const uint32_t numObservations = numFeatures;
    // m_refPatches                   = cv::Mat( numFeatures, m_patchArea, CV_32F );
    m_optimizer.initParameters( numFeatures * 2 );
    m_refVisibility.resize( numFeatures, false );

    Sophus::SE3d absolutePose = frame->m_TransW2F;

    auto lambdaUpdateFunctor = [ this ]( Sophus::SE3d& pose, const Eigen::VectorXd& dx ) -> void { updatePose( pose, dx ); };
    double error             = 0.0;
    Optimizer::Status optimizationStatus;

    // t1 = std::chrono::high_resolution_clock::now();
    computeJacobianPose( frame );
    // timerJacobian += std::chrono::duration_cast< std::chrono::microseconds >( std::chrono::high_resolution_clock::now() - t1
    // ).count();

    auto lambdaResidualFunctor = [ this, &frame ]( Sophus::SE3d& pose ) -> uint32_t { return computeResidualsPose( frame, pose ); };
    // t1 = std::chrono::high_resolution_clock::now();
    std::tie( optimizationStatus, error ) =
      m_optimizer.optimizeGN< Sophus::SE3d >( absolutePose, lambdaResidualFunctor, nullptr, lambdaUpdateFunctor );

    // curFrame->m_TransW2F = refFrame->m_TransW2F * relativePose;
    frame->m_TransW2F = absolutePose;
    Adjustment_Log( DEBUG ) << "Computed Pose: " << frame->m_TransW2F.params().transpose();

    return error;
}

double BundleAdjustment::optimizeStructure( std::shared_ptr< Frame >& frame, const uint32_t maxNumberPoints )
{
    if ( frame->numberObservation() == 0 )
        return 0;

    m_optimizer.setNumUnknowns(3);
    std::vector< std::shared_ptr< Point > > points;
    for ( const auto& feature : frame->m_frameFeatures )
    {
        if ( feature->m_point == nullptr )
        {
            points.push_back( feature->m_point );
        }
    }
    uint32_t setMaxNumerPoints = maxNumberPoints < points.size() ? maxNumberPoints : points.size();
    std::nth_element( points.begin(), points.begin() + maxNumberPoints, points.end(),
                      []( const std::shared_ptr< Point >& lhs, const std::shared_ptr< Point >& rhs ) -> bool {
                          return lhs->m_lastOptimizedTime < rhs->m_lastOptimizedTime;
                      } );

    m_optimizer.m_numUnknowns = 3;
    double error = 0.0;
    for ( uint32_t i( 0 ); i < setMaxNumerPoints; i++ )
    {
        auto& point                   = points[ i ];
        const std::size_t numFeatures = point->m_features.size();
        m_optimizer.initParameters( numFeatures * 2 );
        m_refVisibility.resize( numFeatures, false );

        Sophus::SE3d absolutePose = frame->m_TransW2F;

        auto lambdaUpdateFunctor = [ this ]( std::shared_ptr< Point >& point, const Eigen::Vector3d& dx ) -> void {
            updateStructure( point, dx );
        };
        Optimizer::Status optimizationStatus;

        // t1 = std::chrono::high_resolution_clock::now();
        computeJacobianStructure( point );

        auto lambdaResidualFunctor = [ this ]( std::shared_ptr< Point >& point ) -> uint32_t { return computeResidualsStructure( point ); };
        // t1 = std::chrono::high_resolution_clock::now();
        std::tie( optimizationStatus, error ) =
          m_optimizer.optimizeGN< std::shared_ptr< Point > >( point, lambdaResidualFunctor, nullptr, lambdaUpdateFunctor );
    }

    return error;
}

void BundleAdjustment::computeJacobianPose( const std::shared_ptr< Frame >& frame )
{
    resetParameters();
    const double fx     = frame->m_camera->fx();
    const double fy     = frame->m_camera->fy();
    uint32_t cntFeature = 0;
    uint32_t cntPoints  = 0;
    for ( const auto& feature : frame->m_frameFeatures )
    {
        if ( feature->m_point == nullptr )
        {
            cntFeature++;
            continue;
        }
        m_refVisibility[ cntFeature ] = true;
        const Eigen::Vector3d point   = feature->m_point->m_position;

        Eigen::Matrix< double, 2, 6 > imageJac;
        computeImageJacPose( imageJac, point, fx, fy );

        m_optimizer.m_jacobian.block( 2 * cntPoints, 0, 2, 6 ) = imageJac;
        cntPoints++;
        cntFeature++;
    }
    // visualization::templatePatches( m_refPatches, cntFeature, m_patchSize, 10, 10, 12 );
}

// if we define the residual error as current image - reference image, we do not need to apply the negative for gradient
uint32_t BundleAdjustment::computeResidualsPose( const std::shared_ptr< Frame >& frame, const Sophus::SE3d& pose )
{
    uint32_t cntFeature              = 0;
    uint32_t cntTotalProjectedPixels = 0;
    for ( const auto& feature : frame->m_frameFeatures )
    {
        if ( m_refVisibility[ cntFeature ] == false )
        {
            cntFeature++;
            continue;
        }

        const Eigen::Vector2d error = frame->camera2image( pose * feature->m_point->m_position ) - feature->m_feature;
        // ****
        // IF we compute the error of inverse compositional as r = T(x) - I(W), then we should solve (delta p) = -(JtWT).inverse() *
        // JtWr BUt if we take r = I(W) - T(x) a residual error, then (delta p) = (JtWT).inverse() * JtWr
        // ***

        m_optimizer.m_residuals( cntTotalProjectedPixels++ ) = static_cast< double >( error.x() );
        m_optimizer.m_residuals( cntTotalProjectedPixels )   = static_cast< double >( error.y() );
        // m_optimizer.m_residuals( cntFeature * m_patchArea + cntPixel ) = static_cast< double >( *pixelPtr - curPixelValue);
        m_optimizer.m_visiblePoints( cntTotalProjectedPixels ) = true;

        cntTotalProjectedPixels++;
        cntFeature++;
    }
    return cntTotalProjectedPixels;
}

void BundleAdjustment::computeImageJacPose( Eigen::Matrix< double, 2, 6 >& imageJac,
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

void BundleAdjustment::updatePose( Sophus::SE3d& pose, const Eigen::VectorXd& dx )
{
    pose = pose * Sophus::SE3d::exp( -dx );
}

void BundleAdjustment::computeJacobianStructure( const std::shared_ptr< Point >& point )
{
    uint32_t cntFeature = 0;
    for ( const auto& feature : point->m_features )
    {
        const auto& pos                      = point->m_position;
        // const Eigen::Vector2d projectedPoint = feature->m_frame->world2image( pos );
        const auto& camera                   = feature->m_frame->m_camera;
        Eigen::Matrix< double, 2, 3 > imageJac;
        computeImageJacStructure( imageJac, pos, feature->m_frame->m_TransW2F.rotationMatrix(), camera->fx(), camera->fy() );
        m_optimizer.m_jacobian.block( 2 * cntFeature, 0, 2, 3 ) = imageJac;
        cntFeature++;
    }
}

void BundleAdjustment::computeImageJacStructure(
  Eigen::Matrix< double, 2, 3 >& imageJac, const Eigen::Vector3d& point, const Eigen::Matrix3d& rotation, const double fx, const double fy )
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

    const double x  = point.x();
    const double y  = point.y();
    const double z  = point.z();
    const double z2 = z * z;

    imageJac( 0, 0 ) = fx / z;
    imageJac( 0, 1 ) = 0.0;
    imageJac( 0, 2 ) = -( fx * x ) / z2;

    imageJac( 1, 0 ) = 0.0;
    imageJac( 1, 1 ) = fy / z;
    imageJac( 1, 2 ) = -( fy * y ) / z2;
    imageJac         = imageJac * rotation;
}

uint32_t BundleAdjustment::computeResidualsStructure( const std::shared_ptr< Point >& point )
{
    uint32_t cntFeature = 0;
    for ( const auto& feature : point->m_features )
    {
        const auto& pos                      = point->m_position;
        const Eigen::Vector2d projectedPoint = feature->m_frame->world2image( pos );
        const Eigen::Vector2d error          = projectedPoint - feature->m_feature;
        // ****
        // IF we compute the error of inverse compositional as r = T(x) - I(W), then we should solve (delta p) = -(JtWT).inverse() *
        // JtWr BUt if we take r = I(W) - T(x) a residual error, then (delta p) = (JtWT).inverse() * JtWr
        // ***

        m_optimizer.m_residuals( cntFeature++ ) = static_cast< double >( error.x() );
        m_optimizer.m_residuals( cntFeature )   = static_cast< double >( error.y() );
        // m_optimizer.m_residuals( cntFeature * m_patchArea + cntPixel ) = static_cast< double >( *pixelPtr - curPixelValue);
        m_optimizer.m_visiblePoints( cntFeature ) = true;

        cntFeature++;
    }
    return cntFeature;
}

void BundleAdjustment::updateStructure( const std::shared_ptr< Point >& point, const Eigen::Vector3d& dx )
{
    point->m_position = point->m_position + dx;
}

void BundleAdjustment::resetParameters()
{
    std::fill( m_refVisibility.begin(), m_refVisibility.end(), false );
}
