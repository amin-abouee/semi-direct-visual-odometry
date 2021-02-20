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
#include <map>

#define Adjustment_Log( LEVEL ) CLOG( LEVEL, "Adjustment" )
#define SCHUR_TRICK 1

BundleAdjustment::BundleAdjustment( const std::shared_ptr< PinholeCamera >& camera, int32_t level, uint32_t numParameters )
    : m_camera( camera ), m_level( level ), m_optimizer( numParameters )
{
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
    m_optimizer.initParameters( numFeatures * 3 );
    m_refVisibility.resize( numFeatures, false );

    Sophus::SE3d absolutePose = frame->m_absPose;

    auto lambdaUpdateFunctor = [ this ]( Sophus::SE3d& pose, const Eigen::VectorXd& dx ) -> void { updatePose( pose, dx ); };
    double error             = 0.0;
    Optimizer::Status optimizationStatus;

    // t1 = std::chrono::high_resolution_clock::now();
    // computeJacobianPose( frame );
    auto lambdaJacobianFunctor = [this, &frame] ( Sophus::SE3d& pose ) -> uint32_t { return computeJacobianPose(frame, pose);};
    // timerJacobian += std::chrono::duration_cast< std::chrono::microseconds >( std::chrono::high_resolution_clock::now() - t1
    // ).count();

    auto lambdaResidualFunctor = [ this, &frame ]( Sophus::SE3d& pose ) -> uint32_t { return computeResidualsPose( frame, pose ); };
    // t1 = std::chrono::high_resolution_clock::now();
    std::tie( optimizationStatus, error ) =
      m_optimizer.optimizeLM< Sophus::SE3d >( absolutePose, lambdaResidualFunctor, lambdaJacobianFunctor, lambdaUpdateFunctor );

    // curFrame->m_absPose = refFrame->m_absPose * relativePose;
    frame->m_absPose = absolutePose;
    Adjustment_Log( DEBUG ) << "Computed Pose: " << frame->m_absPose.params().transpose();

    return error;
}

uint32_t BundleAdjustment::computeJacobianPose( const std::shared_ptr< Frame >& frame, const Sophus::SE3d& pose )
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
        const Eigen::Vector3d point   = pose * feature->m_point->m_position;

        Eigen::Matrix< double, 3, 6 > imageJac;
        computeImageJacPose( imageJac, point, fx, fy );

        m_optimizer.m_jacobian.block( 3 * cntPoints, 0, 3, 6 ) = imageJac;
        cntPoints++;
        cntFeature++;
    }
    return cntPoints;
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

        const Eigen::Vector3d error = feature->m_bearingVec - (pose * feature->m_point->m_position).normalized();
        // ****
        // IF we compute the error of inverse compositional as r = T(x) - I(W), then we should solve (delta p) = -(JtWT).inverse() *
        // JtWr BUt if we take r = I(W) - T(x) a residual error, then (delta p) = (JtWT).inverse() * JtWr
        // ***

        m_optimizer.m_residuals( cntTotalProjectedPixels++ ) = std::abs(error.x());
        m_optimizer.m_residuals( cntTotalProjectedPixels++ )   = std::abs(error.y());
        m_optimizer.m_residuals( cntTotalProjectedPixels )   =  std::abs(error.z());
        // m_optimizer.m_residuals( cntFeature * m_patchArea + cntPixel ) = static_cast< double >( *pixelPtr - curPixelValue);
        m_optimizer.m_visiblePoints( cntTotalProjectedPixels ) = true;

        cntTotalProjectedPixels++;
        cntFeature++;
    }
    return cntTotalProjectedPixels;
}

void BundleAdjustment::computeImageJacPose( Eigen::Matrix< double, 3, 6 >& imageJac,
                                            const Eigen::Vector3d& point,
                                            const double fx,
                                            const double fy )
{
    // Image Gradient-based Joint Direct Visual Odometry for Stereo Camera, Eq. 12
    // Taking a Deeper Look at the Inverse Compositional Algorithm, Eq. 28
    // https://github.com/uzh-rpg/rpg_svo/blob/master/svo/include/svo/frame.h, jacobian_xyz2uv function but the negative
    // one

    //                              ⎡1  0  0  0   z   -y⎤
    //                              ⎢                   ⎥
    // dX / d(theta) = [I [X]x]     ⎢0  1  0  -z  0   x ⎥
    //                              ⎢                   ⎥
    //                              ⎣0  0  1  y   -x  0 ⎦

    const double x  = point.x();
    const double y  = point.y();
    const double z  = point.z();

    imageJac (0, 0 ) = 1.0;
    imageJac (0, 1 ) = 0.0;
    imageJac (0, 2 ) = 0.0;
    imageJac (0, 3 ) = 0.0;
    imageJac (0, 4 ) = z;
    imageJac (0, 5 ) = -y;

    imageJac (1, 0 ) = 0.0;
    imageJac (1, 1 ) = 1.0;
    imageJac (1, 2 ) = 0.0;
    imageJac (1, 3 ) = -z;
    imageJac (1, 4 ) = 0.0;
    imageJac (1, 5 ) = x;

    imageJac (2, 0 ) = 0.0;
    imageJac (2, 1 ) = 0.0;
    imageJac (2, 2 ) = 1.0;
    imageJac (2, 3 ) = y;
    imageJac (2, 4 ) = -x;
    imageJac (2, 5 ) = 0.0;
}

void BundleAdjustment::updatePose( Sophus::SE3d& pose, const Eigen::VectorXd& dx )
{
    // pose = pose * Sophus::SE3d::exp( -dx );
    pose = Sophus::SE3d::exp( dx ) * pose;
}

double BundleAdjustment::optimizeStructure( std::shared_ptr< Frame >& frame, const uint32_t maxNumberPoints )
{
    if ( frame->numberObservationWithPoints() == 0 )
        return 0;

    m_optimizer.setNumUnknowns( 3 );
    std::vector< std::shared_ptr< Point > > points;
    for ( const auto& feature : frame->m_features )
    {
        if ( feature->m_point != nullptr )
        {
            points.push_back( feature->m_point );
        }
    }
    uint32_t setMaxNumerPoints = maxNumberPoints < points.size() ? maxNumberPoints : points.size();
    std::nth_element( points.begin(), points.begin() + maxNumberPoints, points.end(),
                      []( const std::shared_ptr< Point >& lhs, const std::shared_ptr< Point >& rhs ) -> bool {
                          return lhs->m_lastOptimizedTime < rhs->m_lastOptimizedTime;
                      } );

    // m_optimizer.m_numUnknowns = 3;
    double error = 0.0;
    for ( uint32_t i( 0 ); i < setMaxNumerPoints; i++ )
    {
        auto& point = points[ i ];
        Adjustment_Log( DEBUG ) << "Point id " << point->m_id;
        Adjustment_Log( DEBUG ) << "Old Position:  " << point->m_position.transpose()
                                << ", error: " << algorithm::computeStructureError( point );
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
        Adjustment_Log( DEBUG ) << "New Position:  " << point->m_position.transpose()
                                << ", error: " << algorithm::computeStructureError( point );
    }

    return error;
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
        const Eigen::Vector2d error          = -projectedPoint + feature->m_pixelPosition;
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
    // https://github.com/RainerKuemmerle/g2o/blob/master/g2o/examples/ba/ba_demo.cpp
    std::unique_ptr< g2o::BlockSolver_6_3::LinearSolverType > linearSolver =
      g2o::make_unique< g2o::LinearSolverCholmod< g2o::BlockSolver_6_3::PoseMatrixType > >();
    g2o::OptimizationAlgorithmLevenberg* solver =
      new g2o::OptimizationAlgorithmLevenberg( g2o::make_unique< g2o::BlockSolver_6_3 >( std::move( linearSolver ) ) );
#else
    std::unique_ptr< g2o::BlockSolver_6_3::LinearSolverType > linearSolver =
      g2o::make_unique< g2o::LinearSolverCholmod< g2o::BlockSolver_6_3::PoseMatrixType > >();
    g2o::OptimizationAlgorithmLevenberg* solver =
      new g2o::OptimizationAlgorithmLevenberg( g2o::make_unique< g2o::BlockSolverX >( std::move( linearSolver ) ) );
#endif

    solver->setMaxTrialsAfterFailure( 5 );
    optimizer.setAlgorithm( solver );

    g2o::CameraParameters* camParams = new g2o::CameraParameters( m_camera->fx(), m_camera->principalPoint(), 0.0 );
    camParams->setId( 0 );
    if ( !optimizer.addParameter( camParams ) )
    {
        Adjustment_Log( ERROR ) << "Camera parameters did not set for solver";
    }
}

BundleAdjustment::g2oFrameSE3* BundleAdjustment::createG2oFrameSE3( const std::shared_ptr< Frame >& frame,
                                                                    const uint32_t id,
                                                                    const bool fixed )
{
    g2oFrameSE3* vertex = new g2oFrameSE3();
    vertex->setId( id );
    vertex->setFixed( fixed );
    const Eigen::Quaternion rot = frame->m_absPose.unit_quaternion();
    g2o::SE3Quat pose( rot, frame->m_absPose.translation() );
    vertex->setEstimate( pose );
    return vertex;
}

BundleAdjustment::g2oPoint* BundleAdjustment::createG2oPoint( const Eigen::Vector3d position, const uint32_t id, const bool fixed )
{
    g2oPoint* vertex = new g2oPoint();
    vertex->setId( id );
#if SCHUR_TRICK
    vertex->setMarginalized( true );
#endif
    vertex->setFixed( fixed );
    vertex->setEstimate( position );
    return vertex;
}

BundleAdjustment::g2oEdgeSE3* BundleAdjustment::createG2oEdgeSE3(
  g2oFrameSE3* v_kf, g2oPoint* v_mp, const Eigen::Vector2d& up, bool robustKernel, double huberWidth, double weight )
{
    g2oEdgeSE3* edge = new g2oEdgeSE3();
    edge->setVertex( 0, dynamic_cast< g2o::OptimizableGraph::Vertex* >( v_mp ) );
    edge->setVertex( 1, dynamic_cast< g2o::OptimizableGraph::Vertex* >( v_kf ) );
    edge->setMeasurement( up );
    edge->information() = weight * Eigen::Matrix2d::Identity( 2, 2 );
    if ( robustKernel == true )
    {
        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber();  // TODO: memory leak
        rk->setDelta( huberWidth );
        edge->setRobustKernel( rk );
    }
    edge->setParameterId( 0, 0 );
    return edge;
}

void BundleAdjustment::runSparseBAOptimizer( g2o::SparseOptimizer& optimizer,
                                             uint32_t numIterations,
                                             double& initError,
                                             double& finalError )
{
    optimizer.initializeOptimization();
    optimizer.computeActiveErrors();
    initError = optimizer.activeRobustChi2();
    optimizer.optimize( numIterations );
    finalError = optimizer.activeRobustChi2();
    Adjustment_Log( DEBUG ) << "Init Error: " << std::sqrt( initError ) << " -> Final Error: " << std::sqrt( finalError );
}

void BundleAdjustment::twoViewBA( std::shared_ptr< Frame >& fstFrame,
                                  std::shared_ptr< Frame >& secFrame,
                                  std::shared_ptr< Map >& map,
                                  const double reprojectionError )
{
    // init g2o
    g2o::SparseOptimizer optimizer;
    setupG2o( optimizer );

    std::vector< EdgeContainerSE3 > edges;
    edges.reserve( 400 );
    uint32_t verticesIdx = 0;

    // New keyframe vertex 1: This keyframe is set to fixed!
    g2oFrameSE3* vertexFrame1 = createG2oFrameSE3( fstFrame, verticesIdx++, true );
    optimizer.addVertex( vertexFrame1 );

    // New keyframe vertex 2
    g2oFrameSE3* vertexFrame2 = createG2oFrameSE3( secFrame, verticesIdx++, false );
    optimizer.addVertex( vertexFrame2 );

    // Create point vertices
    for ( auto& feature : fstFrame->m_features )
    {
        auto& point = feature->m_point;
        if ( point == nullptr )
            continue;
        g2oPoint* vertexPoint = createG2oPoint( point->m_position, verticesIdx++, false );
        optimizer.addVertex( vertexPoint );
        point->m_optG2oPoint = vertexPoint;
        g2oEdgeSE3* edge1    = createG2oEdgeSE3( vertexFrame1, vertexPoint, feature->m_pixelPosition, true, reprojectionError * 1.0 );
        optimizer.addEdge( edge1 );
        edges.push_back(
          EdgeContainerSE3( edge1, fstFrame, feature ) );  // TODO feature now links to frame, so we can simplify edge container!

        // find at which index the second frame observes the point
        auto& featureSecFrame = point->findFeature( secFrame );
        g2oEdgeSE3* edge2 = createG2oEdgeSE3( vertexFrame2, vertexPoint, featureSecFrame->m_pixelPosition, true, reprojectionError * 1.0 );
        optimizer.addEdge( edge2 );
        edges.push_back( EdgeContainerSE3( edge2, secFrame, featureSecFrame ) );
    }

    Adjustment_Log( DEBUG ) << "Graph for optimization has " << optimizer.vertices().size() << " vetices and " << optimizer.edges().size()
                            << " edges";

    // Optimization
    double initError, finalError;
    runSparseBAOptimizer( optimizer, 10, initError, finalError );

    // Update keyframe positions. we don't need to check the first frame
    fstFrame->m_absPose.rotationMatrix() = vertexFrame1->estimate().rotation().toRotationMatrix();
    fstFrame->m_absPose.translation()    = vertexFrame1->estimate().translation();
    secFrame->m_absPose.rotationMatrix() = vertexFrame2->estimate().rotation().toRotationMatrix();
    secFrame->m_absPose.translation()    = vertexFrame2->estimate().translation();

    // Update mappoint positions
    for ( auto& feature : fstFrame->m_features )
    {
        if ( feature->m_point == nullptr )
            continue;
        feature->m_point->m_position    = feature->m_point->m_optG2oPoint->estimate();
        feature->m_point->m_optG2oPoint = nullptr;
    }

    // Find Mappoints with too large reprojection error
    const double chiSquaredError = reprojectionError * reprojectionError;
    uint32_t removedPoints       = 0;
    for ( auto& edgeContainer : edges )
    {
        if ( edgeContainer.edge->chi2() > chiSquaredError )
        {
            if ( edgeContainer.feature->m_point != nullptr )
            {
                // TODO: it is better to call removeFeature
                map->removeFeature( edgeContainer.feature );
                // edgeContainer.feature->m_point = nullptr;
            }
            ++removedPoints;
        }
    }
    Adjustment_Log( DEBUG ) << "number of removed points: " << removedPoints;
}

void BundleAdjustment::localBA( std::shared_ptr< Map >& map, const double reprojectionError )
{
    // init g2o
    g2o::SparseOptimizer optimizer;
    setupG2o( optimizer );

    std::vector< EdgeContainerSE3 > edges;
    std::set< std::shared_ptr< Point > > points;
    std::vector< std::shared_ptr< Frame > > neighborsKeyframe;
    uint32_t verticesIdx = 0;
    uint32_t cntFrames   = 0;
    uint32_t cntPoints   = 0;

    // Add all core keyframes
    for ( auto& keyframe : map->m_keyFrames )
    {
        g2oFrameSE3* vertexFrame = createG2oFrameSE3( keyframe, verticesIdx++, false );
        keyframe->m_optG2oFrame  = vertexFrame;
        optimizer.addVertex( vertexFrame );
        cntFrames++;

        // all points that the core keyframes observe are also optimized:
        for ( auto& feature : keyframe->m_features )
            if ( feature->m_point != nullptr && feature->m_point->numberObservation() > 1 )
                points.insert( feature->m_point );
    }

    // Adjustment_Log( DEBUG ) << "Num Keyframes: " << map->m_keyFrames.size() << ", points: " << points.size();

    // Now go throug all the points and add a measurement. Add a fixed neighbour keyframe if it is not in the set of core kfs
    for ( const auto& point : points )
    {
        if ( point->numberObservation() == 1 )
        {
            Adjustment_Log( WARNING ) << "point " << point->m_id << " has just one observation";
            // map->removePoint(point);
            continue;
        }
        // Create point vertex
        g2oPoint* vertexPoint = createG2oPoint( point->m_position, verticesIdx++, false );
        point->m_optG2oPoint  = vertexPoint;
        optimizer.addVertex( vertexPoint );
        cntPoints++;

        // Add edges
        for ( auto& feature : point->m_features )
        {
            if (feature->m_point == nullptr)
            {
                Adjustment_Log( WARNING ) << "feature has no point";
            }
            // TODO: check to select the best frame
            if ( feature->m_frame->m_optG2oFrame == nullptr )
            {
                // frame does not have a vertex yet -> it belongs to the neighbors kfs and
                // is fixed. create one:
                g2oFrameSE3* vertexFrame        = createG2oFrameSE3( feature->m_frame, verticesIdx++, true );
                feature->m_frame->m_optG2oFrame = vertexFrame;
                optimizer.addVertex( vertexFrame );
                neighborsKeyframe.push_back( feature->m_frame );
            }

            // create edge
            g2oEdgeSE3* edge = createG2oEdgeSE3( feature->m_frame->m_optG2oFrame, vertexPoint, feature->m_pixelPosition, true,
                                                 reprojectionError * 1.0, 1.0 );
            optimizer.addEdge( edge );
            edges.push_back( EdgeContainerSE3( edge, feature->m_frame, feature ) );
        }
    }

    Adjustment_Log( DEBUG ) << "Graph for optimization has " << optimizer.vertices().size() << " vetices and " << optimizer.edges().size()
                            << " edges";
    Adjustment_Log( DEBUG ) << "Num keyframes: " << cntFrames << ", Num points: " << cntPoints
                            << ", Neighbors vertex: " << neighborsKeyframe.size();

    // structure only
    g2o::StructureOnlySolver< 3 > structure_only_ba;
    g2o::OptimizableGraph::VertexContainer optimizedPoints;
    for ( g2o::OptimizableGraph::VertexIDMap::const_iterator it = optimizer.vertices().begin(); it != optimizer.vertices().end(); ++it )
    {
        g2o::OptimizableGraph::Vertex* v = static_cast< g2o::OptimizableGraph::Vertex* >( it->second );
        if ( v->dimension() == 3 && v->edges().size() >= 2 )
            optimizedPoints.push_back( v );
    }
    structure_only_ba.calc( optimizedPoints, 10 );

    // Optimization
    double initError  = 0.0;
    double finalError = 0.0;
    runSparseBAOptimizer( optimizer, 10, initError, finalError );

    // Update Keyframes
    for ( auto& keyframe : map->m_keyFrames )
    {
        keyframe->m_absPose =
          Sophus::SE3d( keyframe->m_optG2oFrame->estimate().rotation(), keyframe->m_optG2oFrame->estimate().translation() );
        keyframe->m_optG2oFrame = nullptr;
    }

    // Off for neighbors frames
    for ( auto& frame : neighborsKeyframe )
        frame->m_optG2oFrame = nullptr;

    // Update Mappoints
    for ( auto& point : points )
    {
        point->m_position    = point->m_optG2oPoint->estimate();
        point->m_optG2oPoint = nullptr;
    }

    // Remove Measurements with too large reprojection error
    double chiSquaredError   = reprojectionError * reprojectionError;
    uint32_t removedFeatures = 0;
    for ( auto& edgeCointer : edges )
    {
        if ( edgeCointer.edge->chi2() > chiSquaredError )
        {
            // TODO: it is better to call removeFeature
            map->removeFeature( edgeCointer.feature );
            removedFeatures++;
        }
    }

    Adjustment_Log( DEBUG ) << "number of removed features: " << removedFeatures;

    // uint32_t removedPoints = 0;
    // for ( auto& point : points )
    // {
    //     if ( point->numberObservation() == 1 )
    //     {
    //         Adjustment_Log( WARNING ) << "point " << point->m_id << " has just one observation";
    //         map->removePoint( point );
    //         removedPoints++;
    //     }
    // }

    // Adjustment_Log( DEBUG ) << "number of removed points: " << removedPoints;
}

void BundleAdjustment::oneFrameWithScene( std::shared_ptr< Frame >& frame, const double reprojectionError )
{
    // init g2o
    g2o::SparseOptimizer optimizer;
    setupG2o( optimizer );

    // std::vector< std::shared_ptr< Point > > points;
    // std::vector< std::shared_ptr< Frame > > neighborsKeyframe;
    uint32_t verticesIdx = 0;
    uint32_t cntFrames   = 0;
    uint32_t cntPoints   = 0;

    // New keyframe vertex 1: This keyframe is set to fixed!
    g2oFrameSE3* vertexFrame = createG2oFrameSE3( frame, verticesIdx++, false );
    frame->m_optG2oFrame     = vertexFrame;
    optimizer.addVertex( vertexFrame );
    cntFrames++;

    for ( auto& feature : frame->m_features )
    {
        if ( feature->m_point != nullptr && feature->m_point->numberObservation() > 1 )
        {
            auto& point = feature->m_point;

            // Create point vertex
            g2oPoint* vertexPoint = createG2oPoint( point->m_position, verticesIdx++, false );
            point->m_optG2oPoint  = vertexPoint;
            optimizer.addVertex( vertexPoint );
            cntPoints++;

            // Add edges
            for ( auto& ptFeature : point->m_features )
            {
                if ( ptFeature->m_point == nullptr )
                {
                    Adjustment_Log( WARNING ) << "feature has no point";
                }

                if ( ptFeature->m_frame->m_optG2oFrame == nullptr )
                {
                    g2oFrameSE3* vertexFrame        = createG2oFrameSE3( ptFeature->m_frame, verticesIdx++, true );
                    ptFeature->m_frame->m_optG2oFrame = vertexFrame;
                    optimizer.addVertex( vertexFrame );
                    cntFrames++;
                }

                // create edge
                g2oEdgeSE3* edge = createG2oEdgeSE3( feature->m_frame->m_optG2oFrame, vertexPoint, ptFeature->m_pixelPosition, true,
                                                     reprojectionError, 1.0 );
                optimizer.addEdge( edge );
                verticesIdx++;
            }
        }
    }

    Adjustment_Log( DEBUG ) << "Number of frames: " << cntFrames << ", Number of points: " << cntPoints
                            << ", Number of vertices: " << verticesIdx;

    Adjustment_Log( DEBUG ) << "Graph for optimization has " << optimizer.vertices().size() << " vertices and " << optimizer.edges().size()
                            << " edges";

    // Optimization
    double initError, finalError;
    runSparseBAOptimizer( optimizer, 10, initError, finalError );

    // Update keyframe positions. we don't need to check the first frame
    frame->m_absPose.rotationMatrix() = frame->m_optG2oFrame->estimate().rotation().toRotationMatrix();
    frame->m_absPose.translation()    = frame->m_optG2oFrame->estimate().translation();
    frame->m_optG2oFrame                   = nullptr;

    // Update mappoint positions
    for ( auto& feature : frame->m_features )
    {
        if ( feature->m_point == nullptr )
            continue;
        feature->m_point->m_position    = feature->m_point->m_optG2oPoint->estimate();
        feature->m_point->m_optG2oPoint = nullptr;
    }

    for ( auto& feature : frame->m_features )
    {
        if ( feature->m_point != nullptr )
        {
            for ( auto& feature : feature->m_point->m_features )
            {
                if ( feature->m_frame->m_optG2oFrame != nullptr )
                {
                    feature->m_frame->m_optG2oFrame = nullptr;
                }
            }
        }
    }
}

void BundleAdjustment::optimizeScene (std::shared_ptr< Frame >& frame, const double reprojectionError)
{
    g2o::SparseOptimizer optimizer;
    setupG2o( optimizer );

    std::vector< EdgeContainerSE3 > edges;
    std::set< std::shared_ptr< Point > > points;
    std::map < uint64_t, g2oFrameSE3*> visitedFrames;
    // std::vector< std::shared_ptr< Frame > > neighborsKeyframe;
    uint32_t verticesIdx = 0;
    uint32_t cntFrames   = 0;
    uint32_t cntPoints   = 0;

    // g2oFrameSE3* vertexMainFrame = createG2oFrameSE3( frame, verticesIdx++, false );
    // frame->m_optG2oFrame  = vertexMainFrame;
    // visitedFrames[frame->m_id] = vertexMainFrame;

    // Add all core keyframes
    // all points that the core keyframes observe are also optimized:
    for ( auto& feature : frame->m_features )
        if ( feature->m_point != nullptr && feature->m_point->numberObservation() > 1 )
            points.insert( feature->m_point );

    Adjustment_Log( DEBUG ) << "Num Points: " << points.size();

    // Adjustment_Log( DEBUG ) << "Num Keyframes: " << map->m_keyFrames.size() << ", points: " << points.size();

    // Now go throug all the points and add a measurement. Add a fixed neighbour keyframe if it is not in the set of core kfs
    for ( const auto& point : points )
    {
        if ( point->numberObservation() == 1 )
        {
            Adjustment_Log( WARNING ) << "point " << point->m_id << " has just one observation";
            // map->removePoint(point);
            continue;
        }
        // Create point vertex
        g2oPoint* vertexPoint = createG2oPoint( point->m_position, verticesIdx++, false );
        point->m_optG2oPoint  = vertexPoint;
        optimizer.addVertex( vertexPoint );
        cntPoints++;

        // Add edges
        for ( auto& feature : point->m_features )
        {
            // if (feature->m_point == nullptr)
            // {
            //     Adjustment_Log( WARNING ) << "feature has no point";
            // }
            // TODO: check to select the best frame
            // if ( feature->m_frame->m_optG2oFrame == nullptr )
            g2oFrameSE3* vertexFrame = nullptr;
            auto it = visitedFrames.find(feature->m_frame->m_id);
            if ( it == visitedFrames.end())
            {
                // frame does not have a vertex yet -> it belongs to the neighbors kfs and
                // is fixed. create one:
                vertexFrame = createG2oFrameSE3( feature->m_frame, verticesIdx++, true );
                // feature->m_frame->m_optG2oFrame = vertexFrame;
                optimizer.addVertex( vertexFrame );
                visitedFrames[feature->m_frame->m_id] = vertexFrame;
                // neighborsKeyframe.push_back( feature->m_frame );
            }
            else
            {
                vertexFrame = it->second;
            }


            // create edge
            g2oEdgeSE3* edge = createG2oEdgeSE3( vertexFrame, vertexPoint, feature->m_pixelPosition, true,
                                                 reprojectionError * 1.0, 1.0 );
            optimizer.addEdge( edge );
            edges.push_back( EdgeContainerSE3( edge, feature->m_frame, feature ) );
        }
        // Adjustment_Log( WARNING ) << "Edge Size: " << edges.size();
    }

    Adjustment_Log( DEBUG ) << "Size Map: " << visitedFrames.size();
    Adjustment_Log( DEBUG ) << "Graph for optimization has " << optimizer.vertices().size() << " vetices and " << optimizer.edges().size()
                            << " edges";

    // structure only
    g2o::StructureOnlySolver< 3 > structure_only_ba;
    g2o::OptimizableGraph::VertexContainer optimizedPoints;
    for ( g2o::OptimizableGraph::VertexIDMap::const_iterator it = optimizer.vertices().begin(); it != optimizer.vertices().end(); ++it )
    {
        g2o::OptimizableGraph::Vertex* v = static_cast< g2o::OptimizableGraph::Vertex* >( it->second );
        if ( v->dimension() == 3 && v->edges().size() >= 2 )
            optimizedPoints.push_back( v );
    }
    structure_only_ba.calc( optimizedPoints, 10 );

    // Optimization
    // double initError  = 0.0;
    // double finalError = 0.0;
    // runSparseBAOptimizer( optimizer, 10, initError, finalError );

    // Update Mappoints
    for ( auto& point : points )
    {
        point->m_position    = point->m_optG2oPoint->estimate();
        point->m_optG2oPoint = nullptr;
    }

    // frame->m_absPose.rotationMatrix() = vertexMainFrame->estimate().rotation().toRotationMatrix();
    // frame->m_absPose.translation()    = vertexMainFrame->estimate().translation();
    // frame->m_optG2oFrame = nullptr;
}


void BundleAdjustment::threeViewBA( std::shared_ptr< Frame >& frame,
                    const double reprojectionError )
{

    // init g2o
    g2o::SparseOptimizer optimizer;
    setupG2o( optimizer );

    // std::vector< EdgeContainerSE3 > edges;
    // edges.reserve( 400 );
    uint32_t verticesIdx = 0;

    // New keyframe vertex 1: This keyframe is set to fixed!
    g2oFrameSE3* vertexFrame1 = createG2oFrameSE3( frame->m_lastKeyframe->m_lastKeyframe, verticesIdx++, true );
    optimizer.addVertex( vertexFrame1 );

    // New keyframe vertex 2
    g2oFrameSE3* vertexFrame2 = createG2oFrameSE3( frame->m_lastKeyframe, verticesIdx++, true );
    optimizer.addVertex( vertexFrame2 );

    // New keyframe vertex 3
    g2oFrameSE3* vertexFrame3 = createG2oFrameSE3( frame, verticesIdx++, false );
    optimizer.addVertex( vertexFrame3 );

    // Create point vertices
    for ( auto& feature : frame->m_features )
    {
        auto& point = feature->m_point;
        if ( point == nullptr )
            continue;
        g2oPoint* vertexPoint = createG2oPoint( point->m_position, verticesIdx++, true );
        optimizer.addVertex( vertexPoint );
        point->m_optG2oPoint = vertexPoint;
        g2oEdgeSE3* edge1    = createG2oEdgeSE3( vertexFrame3, vertexPoint, feature->m_pixelPosition, true, reprojectionError * 1.0 );
        optimizer.addEdge( edge1 );
        // edges.push_back(
        //   EdgeContainerSE3( edge1, fstFrame, feature ) );  // TODO feature now links to frame, so we can simplify edge container!

        // find at which index the second frame observes the point
        auto& featureLastFrame = point->findFeature( frame->m_lastKeyframe );
        if (featureLastFrame != nullptr)
        {
            g2oEdgeSE3* edge2 = createG2oEdgeSE3( vertexFrame2, vertexPoint, featureLastFrame->m_pixelPosition, true, reprojectionError * 1.0 );
            optimizer.addEdge( edge2 );
        }
        // edges.push_back( EdgeContainerSE3( edge2, secFrame, featureSecFrame ) );

        auto& featureSecLastFrame = point->findFeature( frame->m_lastKeyframe->m_lastKeyframe );
        if (featureSecLastFrame != nullptr)
        {
            g2oEdgeSE3* edge2 = createG2oEdgeSE3( vertexFrame1, vertexPoint, featureSecLastFrame->m_pixelPosition, true, reprojectionError * 1.0 );
            optimizer.addEdge( edge2 );
        }
    }

    Adjustment_Log( DEBUG ) << "Graph for optimization has " << optimizer.vertices().size() << " vetices and " << optimizer.edges().size()
                            << " edges";

    // Optimization
    double initError, finalError;
    runSparseBAOptimizer( optimizer, 10, initError, finalError );

    // Update keyframe positions. we don't need to check the first frame
    frame->m_absPose.rotationMatrix() = vertexFrame3->estimate().rotation().toRotationMatrix();
    frame->m_absPose.translation()    = vertexFrame3->estimate().translation();
    // secFrame->m_absPose.rotationMatrix() = vertexFrame2->estimate().rotation().toRotationMatrix();
    // secFrame->m_absPose.translation()    = vertexFrame2->estimate().translation();

    // Update mappoint positions
    // for ( auto& feature : frame->m_features )
    // {
    //     if ( feature->m_point == nullptr )
    //         continue;
    //     feature->m_point->m_position    = feature->m_point->m_optG2oPoint->estimate();
    //     feature->m_point->m_optG2oPoint = nullptr;
    // }

}