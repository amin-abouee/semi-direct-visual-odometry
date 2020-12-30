#include "bundle_adjustment.hpp"
#include "algorithm.hpp"
#include "feature.hpp"
#include "utils.hpp"
#include "visualization.hpp"

#include <easylogging++.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/solver.h>
#include <g2o/solvers/cholmod/linear_solver_cholmod.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/solvers/structure_only/structure_only_solver.h>

#include <algorithm>
#include <deque>
#include <iostream>
#include <set>
#include <vector>

#define Adjustment_Log( LEVEL ) CLOG( LEVEL, "Adjustment" )
#define SCHUR_TRICK 1

BundleAdjustment::BundleAdjustment( int32_t level, uint32_t numParameters ) : m_level( level ), m_optimizer( numParameters )
{
    // el::Loggers::getLogger( "Tracker" );  // Register new logger
    // std::cout << "c'tor image alignment" << std::endl;
}

double BundleAdjustment::optimizePose( std::shared_ptr< Frame >& frame )
{
    if ( frame->numberObservation() == 0 )
        return 0;

    m_optimizer.setNumUnknowns( 6 );
    // auto t1 = std::chrono::high_resolution_clock::now();
    const std::size_t numFeatures = frame->numberObservation();
    // const uint32_t numObservations = numFeatures;
    // m_refPatches                   = cv::Mat( numFeatures, m_patchArea, CV_32F );
    m_optimizer.initParameters( numFeatures * 2 );
    m_refVisibility.resize( numFeatures, false );

    Sophus::SE3d absolutePose = frame->m_absPose;

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

    // curFrame->m_absPose = refFrame->m_absPose * relativePose;
    frame->m_absPose = absolutePose;
    Adjustment_Log( DEBUG ) << "Computed Pose: " << frame->m_absPose.params().transpose();

    return error;
}

double BundleAdjustment::optimizeStructure( std::shared_ptr< Frame >& frame, const uint32_t maxNumberPoints )
{
    if ( frame->numberObservation() == 0 )
        return 0;

    m_optimizer.setNumUnknowns( 3 );
    std::vector< std::shared_ptr< Point > > points;
    for ( const auto& feature : frame->m_features )
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
    double error              = 0.0;
    for ( uint32_t i( 0 ); i < setMaxNumerPoints; i++ )
    {
        auto& point                   = points[ i ];
        const std::size_t numFeatures = point->m_features.size();
        m_optimizer.initParameters( numFeatures * 2 );
        m_refVisibility.resize( numFeatures, false );

        Sophus::SE3d absolutePose = frame->m_absPose;

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
    for ( const auto& feature : frame->m_features )
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
    for ( const auto& feature : frame->m_features )
    {
        if ( m_refVisibility[ cntFeature ] == false )
        {
            cntFeature++;
            continue;
        }

        const Eigen::Vector2d error = frame->camera2image( pose * feature->m_point->m_position ) - feature->m_pixelPosition;
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
        const auto& pos = point->m_position;
        // const Eigen::Vector2d projectedPoint = feature->m_frame->world2image( pos );
        const auto& camera = feature->m_frame->m_camera;
        Eigen::Matrix< double, 2, 3 > imageJac;
        computeImageJacStructure( imageJac, pos, feature->m_frame->m_absPose.rotationMatrix(), camera->fx(), camera->fy() );
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
        const Eigen::Vector2d error          = projectedPoint - feature->m_pixelPosition;
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

void BundleAdjustment::setupG2o( g2o::SparseOptimizer& optimizer )
{
    optimizer.setVerbose( false );

#if SCHUR_TRICK
    // solver
    // https://github.com/RainerKuemmerle/g2o/blob/master/unit_test/slam3d/optimization_slam3d.cpp
    std::unique_ptr< g2o::BlockSolver_6_3::LinearSolverType > linearSolver =
      g2o::make_unique< g2o::LinearSolverCholmod< g2o::BlockSolver_6_3::PoseMatrixType > >();
    g2o::OptimizationAlgorithmLevenberg* solver =
      new g2o::OptimizationAlgorithmLevenberg( g2o::make_unique< g2o::BlockSolver_6_3 >( std::move( linearSolver ) ) );
#else
    g2o::BlockSolverX::LinearSolverType* linearSolver;
    linearSolver = new g2o::LinearSolverCholmod< g2o::BlockSolverX::PoseMatrixType >();
    // linearSolver = new g2o::LinearSolverCSparse<g2o::BlockSolverX::PoseMatrixType>();
    g2o::BlockSolverX* solver_ptr               = new g2o::BlockSolverX( linearSolver );
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg( solver_ptr );
#endif

    solver->setMaxTrialsAfterFailure( 5 );
    optimizer.setAlgorithm( solver );

    // setup camera
    const double fx = 721.5377;
    const Eigen::Vector2d principlePoint( 609.5593, 172.8540 );
    g2o::CameraParameters* cam_params = new g2o::CameraParameters( fx, principlePoint, 0. );
    cam_params->setId( 0 );
    if ( !optimizer.addParameter( cam_params ) )
    {
        assert( false );
    }
}

BundleAdjustment::g2oFrameSE3* BundleAdjustment::createG2oFrameSE3( const std::shared_ptr< Frame >& frame,
                                                                    const uint32_t id,
                                                                    const bool fixed )
{
    g2oFrameSE3* v = new g2oFrameSE3();
    v->setId( id );
    v->setFixed( fixed );
    v->setEstimate( g2o::SE3Quat( frame->m_absPose.unit_quaternion(), frame->m_absPose.translation() ) );
    return v;
}

BundleAdjustment::g2oPoint* BundleAdjustment::createG2oPoint( const Eigen::Vector3d position, const uint32_t id, const bool fixed )
{
    g2oPoint* v = new g2oPoint;
    v->setId( id );
#if SCHUR_TRICK
    v->setMarginalized( true );
#endif
    v->setFixed( fixed );
    v->setEstimate( position );
    return v;
}

BundleAdjustment::g2oEdgeSE3* BundleAdjustment::createG2oEdgeSE3(
  g2oFrameSE3* v_kf, g2oPoint* v_mp, const Eigen::Vector2d& up, bool robustKernel, double huberWidth, double weight )
{
    g2oEdgeSE3* e = new g2oEdgeSE3();
    e->setVertex( 0, dynamic_cast< g2o::OptimizableGraph::Vertex* >( v_mp ) );
    e->setVertex( 1, dynamic_cast< g2o::OptimizableGraph::Vertex* >( v_kf ) );
    e->setMeasurement( up );
    e->information() = weight * Eigen::Matrix2d::Identity( 2, 2 );
    if ( robustKernel == true )
    {
        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber();  // TODO: memory leak
        rk->setDelta( huberWidth );
        e->setRobustKernel( rk );
    }
    e->setParameterId( 0, 0 );
    return e;
}

void BundleAdjustment::runSparseBAOptimizer( g2o::SparseOptimizer& optimizer,
                                             uint32_t numIterations,
                                             double& initError,
                                             double& finalError )
{
    optimizer.initializeOptimization();
    optimizer.computeActiveErrors();
    initError = optimizer.activeChi2();
    optimizer.optimize( numIterations );
    finalError = optimizer.activeChi2();
    Adjustment_Log (DEBUG) << "Init Error: " << initError << " -> Final Error: " << finalError;
}

void BundleAdjustment::twoViewBA( std::shared_ptr< Frame >& fstFrame,
                                  std::shared_ptr< Frame >& secFrame,
                                  double reprojectionError,
                                  std::shared_ptr< Map >& map )
{
    // scale reprojection threshold in pixels to unit plane
    // reprojectionError /= fstFrame->m_camera->fx();

    // init g2o
    g2o::SparseOptimizer optimizer;
    setupG2o( optimizer );

    std::vector< EdgeContainerSE3 > edges;
    size_t v_id = 0;

    // New Keyframe Vertex 1: This Keyframe is set to fixed!
    g2oFrameSE3* v_frame1 = createG2oFrameSE3( fstFrame, v_id++, true );
    optimizer.addVertex( v_frame1 );

    // New Keyframe Vertex 2
    g2oFrameSE3* v_frame2 = createG2oFrameSE3( secFrame, v_id++, false );
    optimizer.addVertex( v_frame2 );

    // Create Point Vertices
    for ( auto& feature : fstFrame->m_features )
    {
        auto& point = feature->m_point;
        if ( point == nullptr )
            continue;
        g2oPoint* v_pt = createG2oPoint( point->m_position, v_id++, false );
        optimizer.addVertex( v_pt );
        point->m_optG2oPoint = v_pt;
        g2oEdgeSE3* e1       = createG2oEdgeSE3( v_frame1, v_pt, feature->m_pixelPosition, true, reprojectionError * 1.0 );
        optimizer.addEdge( e1 );
        edges.push_back(
          EdgeContainerSE3( e1, fstFrame, feature ) );  // TODO feature now links to frame, so we can simplify edge container!

        // find at which index the second frame observes the point
        auto& featureSecFrame = point->findFeature( secFrame );
        e1                    = createG2oEdgeSE3( v_frame2, v_pt, featureSecFrame->m_pixelPosition, true, reprojectionError * 1.0 );
        optimizer.addEdge( e1 );
        edges.push_back( EdgeContainerSE3( e1, secFrame, featureSecFrame ) );
    }

    // Optimization
    double initError, finalError;
    runSparseBAOptimizer( optimizer, 10, initError, finalError );

    // Update Keyframe Positions
    fstFrame->m_absPose.rotationMatrix() = v_frame1->estimate().rotation().toRotationMatrix();
    fstFrame->m_absPose.translation()    = v_frame1->estimate().translation();
    secFrame->m_absPose.rotationMatrix() = v_frame2->estimate().rotation().toRotationMatrix();
    secFrame->m_absPose.translation()    = v_frame2->estimate().translation();

    // Update Mappoint Positions
    for ( auto& feature : fstFrame->m_features )
    {
        if ( feature->m_point == nullptr )
            continue;
        feature->m_point->m_position    = feature->m_point->m_optG2oPoint->estimate();
        feature->m_point->m_optG2oPoint = nullptr;
    }

    // Find Mappoints with too large reprojection error
    const double reproj_thresh_squared = reprojectionError * reprojectionError;
    size_t n_incorrect_edges           = 0;
    for ( auto& edgeContainer : edges )
    {
        if ( edgeContainer.edge->chi2() > reproj_thresh_squared )
        {
            if ( edgeContainer.feature->m_point != nullptr )
            {
                map->removePoint( edgeContainer.feature->m_point );
                edgeContainer.feature->m_point = nullptr;
            }
            ++n_incorrect_edges;
        }
    }
}

void BundleAdjustment::localBA( std::shared_ptr< Frame >& frame,
                                std::shared_ptr< Map >& map,
                                uint32_t& incoreectEdge1,
                                uint32_t& incorrectEdge2,
                                double& initError,
                                double& finalError )
{
    // init g2o
    g2o::SparseOptimizer optimizer;
    setupG2o( optimizer );

    std::vector< EdgeContainerSE3 > edges;
    std::set< std::shared_ptr< Point > > mps;
    std::vector< std::shared_ptr< Frame > > neib_kfs;
    size_t v_id      = 0;
    size_t n_mps     = 0;
    size_t n_fix_kfs = 0;
    size_t n_var_kfs = 1;
    size_t n_edges   = 0;
    incoreectEdge1   = 0;
    incorrectEdge2   = 0;

    // Add all core keyframes
    for ( auto& keyframe : map->m_keyFrames )
    {
        g2oFrameSE3* v_kf = createG2oFrameSE3( keyframe, v_id++, false );
        // TODO: I need to add this one
        keyframe->m_optG2oFrame = v_kf;
        ++n_var_kfs;
        assert( optimizer.addVertex( v_kf ) );

        // all points that the core keyframes observe are also optimized:
        for ( auto& feature : keyframe->m_features )
            if ( feature->m_point != nullptr )
                mps.insert( feature->m_point );
    }

    // Now go throug all the points and add a measurement. Add a fixed neighbour
    // Keyframe if it is not in the set of core kfs
    double reproj_thresh         = 2.0 ;
    double reproj_thresh_1_squared = reproj_thresh * reproj_thresh;
    for ( auto& point : mps )
    {
        // Create point vertex
        g2oPoint* v_pt = createG2oPoint( point->m_position, v_id++, false );
        // TODO: add g2o vertex to point
        point->m_optG2oPoint = v_pt;
        assert( optimizer.addVertex( v_pt ) );
        ++n_mps;

        // Add edges
        for ( auto& feature : point->m_features )
        {
            // TODO: double check this line
            Eigen::Vector2d error = feature->m_pixelPosition - feature->m_frame->world2image( point->m_position );

            if ( feature->m_frame->m_optG2oFrame == nullptr )
            {
                // frame does not have a vertex yet -> it belongs to the neib kfs and
                // is fixed. create one:
                g2oFrameSE3* v_kf               = createG2oFrameSE3( feature->m_frame, v_id++, true );
                feature->m_frame->m_optG2oFrame = v_kf;
                ++n_fix_kfs;
                assert( optimizer.addVertex( v_kf ) );
                neib_kfs.push_back( feature->m_frame );
            }

            // create edge
            g2oEdgeSE3* e =
              createG2oEdgeSE3( feature->m_frame->m_optG2oFrame, v_pt, feature->m_pixelPosition, true, reproj_thresh * 1.0, 1.0 );
            assert( optimizer.addEdge( e ) );
            edges.push_back( EdgeContainerSE3( e, feature->m_frame, feature ) );
            ++n_edges;
        }
    }

    // structure only
    g2o::StructureOnlySolver< 3 > structure_only_ba;
    g2o::OptimizableGraph::VertexContainer points;
    for ( g2o::OptimizableGraph::VertexIDMap::const_iterator it = optimizer.vertices().begin(); it != optimizer.vertices().end(); ++it )
    {
        g2o::OptimizableGraph::Vertex* v = static_cast< g2o::OptimizableGraph::Vertex* >( it->second );
        if ( v->dimension() == 3 && v->edges().size() >= 2 )
            points.push_back( v );
    }
    structure_only_ba.calc( points, 10 );

    // Optimization
    // if ( 10 > 0 )
    runSparseBAOptimizer( optimizer, 10, initError, finalError );

    // Update Keyframes
    for (auto& keyframe : map->m_keyFrames)
    {
        keyframe->m_absPose = Sophus::SE3d( keyframe->m_optG2oFrame->estimate().rotation(), keyframe->m_optG2oFrame->estimate().translation() );
        keyframe->m_optG2oFrame = nullptr;
    }

    for ( auto& frame : neib_kfs )
        frame->m_optG2oFrame = nullptr;

    // Update Mappoints
    for ( auto& point : mps )
    {
        point->m_position    = point->m_optG2oPoint->estimate();
        point->m_optG2oPoint = nullptr;
    }

    // Remove Measurements with too large reprojection error
    double reproj_thresh_2_squared = reproj_thresh * reproj_thresh;
    for ( auto& edgeCointer : edges )
    {
        if ( edgeCointer.edge->chi2() > reproj_thresh_2_squared )  //*(1<<it->feature_->level))
        {
            map->removePoint( edgeCointer.feature->m_point );
            ++incorrectEdge2;
        }
    }

    // TODO: delete points and edges!
    initError  = sqrt( initError ) * frame->m_camera->fx();
    finalError = sqrt( finalError ) * frame->m_camera->fx();
}