#include "nlls.hpp"
#include "algorithm.hpp"

NLLS::NLLS( const u_int32_t numUnknowns )
    : m_numUnknowns( numUnknowns )
    , m_hessian( numUnknowns, numUnknowns )
    , m_gradient( numUnknowns )
    , m_dx( numUnknowns )
    , m_maxIteration( 20 )
{
    m_minChiSquaredError = 1e-1;
    m_stepSize           = 1e-16;
    m_normInfDiff        = 1e-3;
    m_maxCoffDx          = 1e+3;
    m_hessian.setZero();
    m_gradient.setZero();
    m_dx.setZero();
    std::cout << "Size hessian: " << m_hessian.rows() << " , " << m_hessian.cols() << std::endl;
    std::cout << "Size gradient: " << m_gradient.size() << std::endl;
}

// double NLLS::optimizeGN( Sophus::SE3d& pose,
//                 const std::function< unsigned int( Sophus::SE3d& pose) >& lambdaResidualFunctor,
//                 const std::size_t numObservations)
// {
//     return optimizeGN(pose, lambdaResidualFunctor, nullptr, numObservations, false);
// }

// double NLLS::optimizeGN( Sophus::SE3d& pose,
//                 const std::function< unsigned int( Sophus::SE3d& pose) >& lambdaResidualFunctor,
//                 const std::function< unsigned int( Sophus::SE3d& pose ) >& lambdaJacobianFunctor,
//                 const std::size_t numObservations)
// {
//     return optimizeGN(pose, lambdaResidualFunctor, lambdaJacobianFunctor, numObservations, true);
// }

double NLLS::optimizeGN( Sophus::SE3d& pose,
                         const std::function< uint32_t( Sophus::SE3d& pose ) >& lambdaResidualFunctor,
                         const std::function< uint32_t( Sophus::SE3d& pose ) >& lambdaJacobianFunctor,
                         const std::function< void( Sophus::SE3d& pose, const Eigen::VectorXd& dx ) >& lambdaUpdateFunctor,
                         const std::size_t numObservations )
{
    const uint32_t numUnknowns = 6;
    // const uint32_t patchSize = numObservations / curVisibility.size();

    assert( ( numObservations - numUnknowns ) > 0 );

    m_residuals.setZero();
    m_weights.setZero();
    m_hessian.setZero();
    m_gradient.setZero();

    bool computeJacobian = lambdaJacobianFunctor == nullptr ? false : true;

    if ( computeJacobian == true )
        m_jacobian.setZero();

    unsigned int curIteration = 0;
    bool stop                 = false;

    double chiSquaredError = 0.0;
    double squaredError    = 0.0;

    double stepSize = 0.0;
    // double normInfDiffPose					= 0.0;
    double normDiffPose              = 0.0;
    uint32_t cntTotalProjectedPixels = 0;
    // unsigned int goodProjectedEdgesJacobian = 0;

    while ( curIteration < m_maxIteration && !stop )
    {
        cntTotalProjectedPixels = lambdaResidualFunctor( pose );
        tukeyWeighting( cntTotalProjectedPixels );
        // const uint32_t validpatches = std::count( curVisibility.begin(), curVisibility.end(), true );
        std::cout << "projected points: " << cntTotalProjectedPixels << std::endl;
        if ( computeJacobian == true )
            lambdaJacobianFunctor( pose );

        for ( std::size_t i( 0 ); i < numObservations; i++ )
        {
            if ( m_visiblePoints( i ) == true )
            {
                const auto Jac = m_jacobian.row( i );
                // std::cout << "Jac " << i << ": " << Jac << std::endl;
                m_hessian.noalias() += Jac.transpose() * Jac * m_weights( i );
                m_gradient.noalias() += Jac.transpose() * m_residuals( i ) * m_weights( i );
                chiSquaredError = m_residuals( i ) * m_residuals( i ) * m_weights( i );
            }
        }
        m_dx.noalias() = m_hessian.ldlt().solve( -m_gradient );

        if ( m_dx.maxCoeff() > m_maxCoffDx || std::isnan( m_dx.cwiseAbs().minCoeff() ) )
            break;

        stepSize     = m_dx.transpose() * m_dx;
        normDiffPose = ( m_dx ).norm() / ( pose.log() ).norm();
        if ( stepSize < m_stepSize || normDiffPose < m_normInfDiff || chiSquaredError < m_minChiSquaredError )
        {
            stop = true;
            // pose = Sophus::SE3d::exp( m_dx ) * pose;
            lambdaUpdateFunctor( pose, m_dx );
            break;
        }
        else
        {
            // pose = Sophus::SE3d::exp( m_dx ) * pose;
            lambdaUpdateFunctor( pose, m_dx );
        }
        ++curIteration;
    }
    return std::sqrt( chiSquaredError / numObservations );
}

double NLLS::optimizeLM( Sophus::SE3d& pose,
                         const std::function< uint32_t( Sophus::SE3d&, Eigen::VectorXd& res ) >& lambdaResidualFunctor,
                         const std::function< uint32_t( Sophus::SE3d&, Eigen::MatrixXd& jac ) >& lambdaJacobianFunctor,
                         const std::function< void( Sophus::SE3d& pose, const Eigen::VectorXd& dx ) >& lambdaUpdateFunctor,
                         const std::size_t numObservations )
{
    return 0.0;
}

void NLLS::initParameters( const std::size_t numObservations )
{
    m_jacobian.resize( numObservations, m_numUnknowns );
    m_residuals.resize( numObservations );
    m_weights.resize( numObservations );
    m_visiblePoints.resize( numObservations );
    m_residuals.setConstant( std::numeric_limits< double >::max() );
    m_weights.setConstant( 0.0 );
    m_visiblePoints.setConstant( false );
}

void NLLS::tukeyWeighting( const uint32_t numValidProjectedPoints )
{
    double sigma = algorithm::computeSigma( m_residuals, numValidProjectedPoints );
    if ( sigma <= std::numeric_limits< double >::epsilon() )
        sigma = std::numeric_limits< double >::epsilon();
    const double c                    = 4.6851 * sigma;
    const double c2                   = c * c;
    const std::size_t numObservations = m_visiblePoints.size();
    for ( std::size_t i( 0 ); i < numObservations; i++ )
    {
        if ( m_visiblePoints( i ) == true )
        {
            double abs = std::abs( m_residuals( i ) );
            if ( abs <= c )
                m_weights( i ) = std::pow( 1.0 - ( ( m_residuals( i ) * m_residuals( i ) ) / c2 ), 2 );
            else
                m_weights( i ) = 0;
        }
    }
}