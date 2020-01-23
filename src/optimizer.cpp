#include "optimizer.hpp"
#include "algorithm.hpp"
#include "utils.hpp"
#include "visualization.hpp"

#include <any>
#include <opencv2/core/eigen.hpp>

Optimizer::Optimizer( const u_int32_t numUnknowns )
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
    m_levenbergMethod = LevenbergMethod::Nielsen;
    // std::cout << "Size hessian: " << m_hessian.rows() << " , " << m_hessian.cols() << std::endl;
    // std::cout << "Size gradient: " << m_gradient.size() << std::endl;

    // m_timerResiduals        = 0;
    // m_timerSolve            = 0;
    // m_timerHessian          = 0;
    // m_timerSwitch           = 0;
    // m_timerLambda            = 0;
    // m_timerUpdateParameters = 0;
    // m_timerCheck            = 0;
    // m_timerFor              = 0;
}

// double Optimizer::optimizeGN( Sophus::SE3d& pose,
//                 const std::function< unsigned int( Sophus::SE3d& pose) >& lambdaResidualFunctor,
//                 const std::size_t numObservations)
// {
//     return optimizeGN(pose, lambdaResidualFunctor, nullptr, numObservations, false);
// }

// double Optimizer::optimizeGN( Sophus::SE3d& pose,
//                 const std::function< unsigned int( Sophus::SE3d& pose) >& lambdaResidualFunctor,
//                 const std::function< unsigned int( Sophus::SE3d& pose ) >& lambdaJacobianFunctor,
//                 const std::size_t numObservations)
// {
//     return optimizeGN(pose, lambdaResidualFunctor, lambdaJacobianFunctor, numObservations, true);
// }

Optimizer::OptimizerResult Optimizer::optimizeGN(
  Sophus::SE3d& pose,
  const std::function< uint32_t( Sophus::SE3d& pose ) >& lambdaResidualFunctor,
  const std::function< uint32_t( Sophus::SE3d& pose ) >& lambdaJacobianFunctor,
  const std::function< void( Sophus::SE3d& pose, const Eigen::VectorXd& dx ) >& lambdaUpdateFunctor )
{
    // const uint32_t numUnknowns     = 6;
    const auto numObservations = m_residuals.size();

    Status optimizeStatus = Status::Failed;

    if ( numObservations < m_numUnknowns )
        return std::make_pair( Status::Non_Suff_Points, -1 );
    // assert( ( numObservations - numUnknowns ) > 0 );

    bool computeJacobian = lambdaJacobianFunctor == nullptr ? false : true;

    unsigned int curIteration = 0;
    // bool stop                        = false;
    double chiSquaredError           = 0.0;
    double stepSize                  = 0.0;
    double normDiffPose              = 0.0;
    uint32_t cntTotalProjectedPixels = 0;

    double preChiSquaredError = std::numeric_limits< double >::max();
    Sophus::SE3d prePose      = pose;

    // while ( curIteration < m_maxIteration && !stop )
    while ( curIteration < m_maxIteration )
    {
        // std::cout << "pose: " << pose.params().format(utils::eigenFormat()) << std::endl;
        resetAllParameters( computeJacobian );
        cntTotalProjectedPixels = lambdaResidualFunctor( pose );
        tukeyWeighting( cntTotalProjectedPixels );
        // const uint32_t validpatches = std::count( curVisibility.begin(), curVisibility.end(), true );
        // std::cout << "projected points: " << cntTotalProjectedPixels << std::endl;
        if ( computeJacobian == true )
            lambdaJacobianFunctor( pose );

        // for ( std::size_t i( 0 ); i < numObservations; i++ )
        // {
        //     if ( m_visiblePoints( i ) == true )
        //     {
        //         const auto Jac = m_jacobian.row( i );
        //         // std::cout << "Jac " << i << ": " << Jac << std::endl;
        //         m_hessian.noalias() += Jac.transpose() * Jac * m_weights( i );
        //         m_gradient.noalias() += Jac.transpose() * m_residuals( i ) * m_weights( i );
        //         chiSquaredError += m_residuals( i ) * m_residuals( i ) * m_weights( i );
        //         // squaredError += m_residuals( i ) * m_residuals( i );
        //     }
        // }
        m_hessian.noalias() = m_jacobian.transpose() * m_weights.asDiagonal() * m_jacobian;
        m_gradient.noalias() = m_jacobian.transpose() * m_weights.asDiagonal() * m_residuals;
        m_dx.noalias() = m_hessian.ldlt().solve( m_gradient );
        // m_dx.noalias() = m_hessian.ldlt().solve( m_gradient );
        // std::cout << "With -gradient: " << (m_hessian.ldlt().solve( -m_gradient )).transpose() << std::endl;
        // std::cout << "With -dx: " << -m_dx.transpose() << std::endl;

        if ( m_dx.maxCoeff() > m_maxCoffDx )
        {
            optimizeStatus = Status::Max_Coff_Dx;
            break;
        }

        if ( std::isnan( m_dx.cwiseAbs().minCoeff() ) )
        {
            optimizeStatus = Status::Non_In_Dx;
            break;
        }

        if ( chiSquaredError > preChiSquaredError )
        {
            optimizeStatus = Status::Increase_Chi_Squred_Error;
            pose           = prePose;  // rollback to previous pose
            break;
        }

        prePose            = pose;
        preChiSquaredError = chiSquaredError;
        std::cout << "chi error: " << chiSquaredError << std::endl;

        stepSize     = m_dx.transpose() * m_dx;
        normDiffPose = ( m_dx ).norm() / ( pose.log() ).norm();

        if ( stepSize < m_stepSize || normDiffPose < m_normInfDiff || chiSquaredError < m_minChiSquaredError )
        {
            // stop = true;
            lambdaUpdateFunctor( pose, m_dx );

            optimizeStatus = stepSize < m_stepSize ? Status::Small_Step_Size : optimizeStatus;
            optimizeStatus = normDiffPose < m_normInfDiff ? Status::Norm_Inf_Diff : optimizeStatus;
            optimizeStatus = chiSquaredError < m_minChiSquaredError ? Status::Small_Chi_Squred_Error : optimizeStatus;
            break;
        }
        else
        {
            lambdaUpdateFunctor( pose, m_dx );
            optimizeStatus = Status::Success;
        }

        visualize( cntTotalProjectedPixels );
        ++curIteration;
    }
    const double rmse = std::sqrt( chiSquaredError / cntTotalProjectedPixels );
    return std::make_pair( optimizeStatus, rmse );
}

Optimizer::OptimizerResult Optimizer::optimizeLM(
  Sophus::SE3d& pose,
  const std::function< uint32_t( Sophus::SE3d& ) >& lambdaResidualFunctor,
  const std::function< uint32_t( Sophus::SE3d& ) >& lambdaJacobianFunctor,
  const std::function< void( Sophus::SE3d& pose, const Eigen::VectorXd& dx ) >& lambdaUpdateFunctor )
{
    // auto t3 = std::chrono::high_resolution_clock::now();
    // const uint32_t numUnknowns     = 6;
    const auto numObservations = m_residuals.size();

    Status optimizeStatus = Status::Failed;
    if ( numObservations < m_numUnknowns )
        return std::make_pair( Status::Non_Suff_Points, -1 );

    bool computeJacobian = lambdaJacobianFunctor == nullptr ? false : true;

    unsigned int curIteration = 0;
    // bool stop                 = false;

    double stepSize     = 0.0;
    double normDiffPose = 0.0;

    double chiSquaredError    = 0.0;
    double preChiSquaredError = 0.0;

    uint32_t cntTotalProjectedPixels = 0;
    uint32_t preTotalProjectedPixels = 0;

    double lambda = 1e-2;
    double nu     = 2.0;

    // auto t1 = std::chrono::high_resolution_clock::now();
    /// run for the first time to get the chi error
    resetAllParameters( computeJacobian );
    cntTotalProjectedPixels = lambdaResidualFunctor( pose );
    tukeyWeighting( cntTotalProjectedPixels );
    chiSquaredError = computeChiSquaredError();
    // m_timerResiduals += std::chrono::duration_cast< std::chrono::microseconds >( std::chrono::high_resolution_clock::now() - t1 ).count();

    Sophus::SE3d prePose = pose;

    Eigen::VectorXd preResiduals( numObservations );
    Eigen::VectorXd preWeights( numObservations );
    Eigen::Matrix< bool, Eigen::Dynamic, 1 > preVisiblePoints( numObservations );

    bool successIteration = true;

    // while ( curIteration < m_maxIteration && !stop )
    while ( curIteration < m_maxIteration )
    {
        // t1 = std::chrono::high_resolution_clock::now();
        // store the current data into the previous ones before updating the parameters
        if ( successIteration == true )
        {
            prePose                 = pose;
            preChiSquaredError      = chiSquaredError;
            preResiduals            = m_residuals;
            preWeights              = m_weights;
            preVisiblePoints        = m_visiblePoints;
            preTotalProjectedPixels = cntTotalProjectedPixels;
            optimizeStatus          = Status::Success;
        }
        // m_timerSwitch += std::chrono::duration_cast< std::chrono::microseconds >( std::chrono::high_resolution_clock::now() - t1 ).count();

        if ( computeJacobian == true )
            lambdaJacobianFunctor( pose );

        // t1 = std::chrono::high_resolution_clock::now();

        // double sumRes = 0.0;
        // double sumWei = 0.0;
        // int cnt = 0;
        // for ( std::size_t i( 0 ); i < numObservations; i++ )
        // {
        //     if ( m_visiblePoints( i ) == true )
        //     {
        //         // cnt++;
        //         // sumRes+= m_residuals(i);
        //         // sumWei+= m_weights(i);
        //         std::cout << "[ "<< i << ", true, "<<m_weights(i)<<", " << m_residuals(i) << ", "<< m_jacobian.row( i ) << " ]" << std::endl;
        //     }
        //     else
        //     {
        //         std::cout << "[ "<< i << ", false, "<<m_weights(i)<<", " << m_residuals(i) << ", "<< m_jacobian.row( i ) << " ]" << std::endl;
        //     }
            
        // }
        // std::cout << std::endl;

        // std::cout << "mean res: " << sumRes/cnt << ". mean weights: " << sumWei/cnt << std::endl;
        // m_hessian.setZero();
        // m_gradient.setZero();
        // for ( std::size_t i( 0 ); i < numObservations; i++ )
        // {
        //     if ( m_visiblePoints( i ) == true )
        //     {
        //         const auto Jac = m_jacobian.row( i );
        //         m_hessian.noalias() += Jac.transpose() * Jac * m_weights( i );
        //         m_gradient.noalias() += Jac.transpose() * m_residuals( i ) * m_weights( i );
                
        //     }
        // }
        // std::cout << "hessian sum: " << m_hessian << std::endl;
        m_hessian.noalias() = m_jacobian.transpose() * m_weights.asDiagonal() * m_jacobian;
        m_gradient.noalias() = m_jacobian.transpose() * m_weights.asDiagonal() * m_residuals;
        // std::cout << "hessian linear: " << m_hessian << std::endl;
        // m_timerHessian += std::chrono::duration_cast< std::chrono::microseconds >( std::chrono::high_resolution_clock::now() - t1 ).count();

        // t1 = std::chrono::high_resolution_clock::now();
        const Eigen::Matrix< double, 1, 6 > jwj = ( m_hessian ).diagonal();

        if ( m_levenbergMethod == LevenbergMethod::Marquardt )
        {
            for ( std::size_t i( 0 ); i < m_numUnknowns; i++ )
                m_hessian( i, i ) += lambda * jwj( i );
        }

        else if ( m_levenbergMethod == LevenbergMethod::Nielsen || m_levenbergMethod == LevenbergMethod::Quadratic )
        {
            if ( curIteration == 0 )
            {
                lambda *= jwj.maxCoeff();
            }
            for ( std::size_t i( 0 ); i < m_numUnknowns; i++ )
                m_hessian( i, i ) += lambda;
        }
        // m_timerLambda += std::chrono::duration_cast< std::chrono::microseconds >( std::chrono::high_resolution_clock::now() - t1 ).count();

        // t1 = std::chrono::high_resolution_clock::now();
        m_dx.noalias() = m_hessian.ldlt().solve( m_gradient );
        // std::cout << "dx: " << m_dx.transpose() << std::endl;
        // pose updated here
        lambdaUpdateFunctor( pose, m_dx );
        // m_timerSolve += std::chrono::duration_cast< std::chrono::microseconds >( std::chrono::high_resolution_clock::now() - t1 ).count();

        // t1 = std::chrono::high_resolution_clock::now();
        if ( m_dx.maxCoeff() > m_maxCoffDx )
        {
            optimizeStatus = Status::Max_Coff_Dx;
            break;
        }

        if ( std::isnan( m_dx.cwiseAbs().minCoeff() ) )
        {
            optimizeStatus = Status::Non_In_Dx;
            break;
        }

        stepSize     = m_dx.transpose() * m_dx;
        normDiffPose = ( m_dx ).norm() / ( prePose.log() ).norm();
        if ( stepSize < m_stepSize || lambda >= 1e14 || lambda <= 1e-14 || normDiffPose < m_normInfDiff )
        {
            optimizeStatus = stepSize < m_stepSize ? Status::Small_Step_Size : optimizeStatus;
            optimizeStatus = std::abs( lambda ) >= 1e14 ? Status::Lambda_Value : optimizeStatus;
            optimizeStatus = normDiffPose < m_normInfDiff ? Status::Norm_Inf_Diff : optimizeStatus;
            break;
        }
        // m_timerCheck += std::chrono::duration_cast< std::chrono::microseconds >( std::chrono::high_resolution_clock::now() - t1 ).count();

        // t1 = std::chrono::high_resolution_clock::now();
        resetResidualParameters();
        cntTotalProjectedPixels = lambdaResidualFunctor( pose );
        tukeyWeighting( cntTotalProjectedPixels );
        chiSquaredError = computeChiSquaredError();
        m_timerResiduals +=
        //   std::chrono::duration_cast< std::chrono::microseconds >( std::chrono::high_resolution_clock::now() - t1 ).count();

        // t1               = std::chrono::high_resolution_clock::now();
        successIteration = updateParameters( preChiSquaredError, chiSquaredError, lambda, nu );
        // m_timerUpdateParameters +=
        //   std::chrono::duration_cast< std::chrono::microseconds >( std::chrono::high_resolution_clock::now() - t1 ).count();

        // t1 = std::chrono::high_resolution_clock::now();
        // rollback to the previous stable parameters
        if ( successIteration == false )
        {
            chiSquaredError         = preChiSquaredError;
            pose                    = prePose;
            m_residuals             = preResiduals;
            m_weights               = preWeights;
            m_visiblePoints         = preVisiblePoints;
            cntTotalProjectedPixels = preTotalProjectedPixels;
        }
        // m_timerSwitch += std::chrono::duration_cast< std::chrono::microseconds >( std::chrono::high_resolution_clock::now() - t1 ).count();
        // std::cout << "Iteration: " << curIteration << ", Chi error: " << chiSquaredError << std::endl;
        // visualize( cntTotalProjectedPixels );
        ++curIteration;
    }
    const double rmse = std::sqrt( chiSquaredError / cntTotalProjectedPixels );
    // m_timerFor += std::chrono::duration_cast< std::chrono::microseconds >( std::chrono::high_resolution_clock::now() - t3 ).count();
    // std::cout << "Pose: " << pose.params().transpose() << std::endl;
    return std::make_pair( optimizeStatus, rmse );
}

void Optimizer::initParameters( const std::size_t numObservations )
{
    m_jacobian.resize( numObservations, m_numUnknowns );
    m_residuals.resize( numObservations );
    m_weights.resize( numObservations );
    m_visiblePoints.resize( numObservations );
}

void Optimizer::resetAllParameters( bool clearJacobian )
{
    m_hessian.setZero();
    m_gradient.setZero();
    m_residuals.setConstant( std::numeric_limits< double >::max() );
    m_weights.setConstant( 0.0 );
    m_visiblePoints.setConstant( false );
    if ( clearJacobian == true )
        m_jacobian.setZero();
}

void Optimizer::resetResidualParameters()
{
    m_residuals.setConstant( std::numeric_limits< double >::max() );
    m_weights.setConstant( 0.0 );
    m_visiblePoints.setConstant( false );
}

bool Optimizer::updateParameters( const double preSquaredError, const double curSquaredError, double& lambda, double& nu )
{
    const double rho = preSquaredError - curSquaredError;
    if ( m_levenbergMethod == LevenbergMethod::Marquardt )
    {
        // Reference levenberg-marquardt equation (16)
        // std::cout << "diff SQ: " << diffSquaredError << std::endl;
        // std::cout << "denominator: " << dx.transpose() * ( lambda * diagHessian * dx + g ) << std::endl;
        // gainRatio = diffSquaredError / ( dx.transpose() * ( lambda * (jwj.transpose() * dx) + g ) );
        // if ( gainRatio > 1e-1 )
        if ( rho > 0.0 )
        {
            lambda = std::max< double >( lambda / 9.0, double( 1e-7 ) );
            return true;
        }
        else
        {
            lambda = std::min< double >( lambda * 11.0, double( 1e7 ) );
            return false;
        }
    }

    // else if ( m_levenbergMethod == LevenbergMethod::Quadratic )
    // {
    //     // Reference levenberg-marquardt equation (15)
    //     // const double gainRatio         = rho / ( m_dx.transpose() * ( lambda * m_dx + g ) );
    //     const double gainRatio         = rho / ( m_dx.transpose() * ( lambda * m_dx ) );
    //     const double gTdx = g.transpose() * m_dx;
    //     double alpha      = gTdx / ( diffSquaredError / 2.0 + 2 * gTdx );
    //     // if ( gainRatio > 1e-1 )
    //     if ( diffSquaredError > 0.0 )
    //     {
    //         curPose = Sophus::SE3d::exp( dx * alpha ) * prePose;
    //         lambda  = std::max< double >( lambda / ( 1 + alpha ), double( 1e-7 ) );
    //         return true;
    //     }
    //     else
    //     {
    //         curPose = prePose;
    //         lambda  = lambda + ( std::fabs( diffSquaredError ) / ( 2 * alpha ) );
    //         return false;
    //     }
    // }

    else if ( m_levenbergMethod == LevenbergMethod::Nielsen )
    {
        // eference levenberg-marquardt equation (15)
        // gainRatio = diffSquaredError / ( dx.transpose() * ( lambda * dx + g ) );
        // if ( gainRatio > 1e-1 )
        if ( rho > 0.0 )
        {
            lambda *= std::max< double >( 1.0 / 3.0, 1.0 - std::pow( ( 2 * rho - 1 ), 3 ) );
            nu = 2.0;
            return true;
        }
        else
        {
            lambda *= nu;
            nu *= 2;
            return false;
        }
    }
    return false;
}

double Optimizer::computeChiSquaredError()
{
    const auto numObservations = m_residuals.size();
    double chiSquaredError     = 0.0;

    for ( std::size_t i( 0 ); i < numObservations; i++ )
    {
        if ( m_visiblePoints( i ) == true )
        {
            chiSquaredError += m_residuals( i ) * m_residuals( i ) * m_weights( i );
        }
    }
    return chiSquaredError;
}

void Optimizer::tukeyWeighting( const uint32_t numValidProjectedPoints )
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
            {
                m_weights( i ) = std::pow( 1.0 - ( ( m_residuals( i ) * m_residuals( i ) ) / c2 ), 2 );
            }
            else
            {
                m_weights( i ) = 0;
            }
        }
    }
    // cv::Mat histogramWeights;
    // visualization::drawHistogram(weights, histogramWeights, 100, 1200, 800, "histo_weights");
    // cv::Mat histogramResiduals;
    // visualization::drawHistogram(residuals, histogramResiduals, 100, 1200, 800, "histo_residuals");
    // visualization::drawHistogram(weights, "g", "weights");
    // visualization::drawHistogram(residuals, "b", "residuals");
}

void Optimizer::visualize( const uint32_t numValidProjectedPoints )
{
    std::map< std::string, std::any > pack;

    std::vector< double > weights;
    std::vector< double > residuals;
    double sum                 = 0.0;
    const auto numObservations = m_visiblePoints.size();
    for ( std::size_t i( 0 ); i < numObservations; i++ )
    {
        if ( m_visiblePoints( i ) == true )
        {
            residuals.push_back( m_residuals( i ) );
            weights.push_back( m_weights( i ) );
            sum += m_residuals( i );
        }
    }

    pack[ "figure_size_width" ]  = uint32_t( 1600 );
    pack[ "figure_size_height" ] = uint32_t( 900 );

    // std::cout << "sum: " << sum << std::endl;

    double median                    = algorithm::computeMedian( m_residuals, numValidProjectedPoints );
    double mad                       = algorithm::computeMAD( m_residuals, numValidProjectedPoints );
    double sigma                     = algorithm::computeSigma( m_residuals, numValidProjectedPoints );
    pack[ "residuals_data" ]         = residuals;
    pack[ "residuals_color" ]        = std::string( "slategray" );
    pack[ "residuals_number_bins" ]  = uint32_t( 50 );
    pack[ "residuals_median" ]       = median;
    pack[ "residuals_median_color" ] = std::string( "royalblue" );
    pack[ "residuals_mad" ]          = mad;
    pack[ "residuals_mad_color" ]    = std::string( "gold" );
    pack[ "residuals_sigma" ]        = sigma;
    pack[ "residuals_sigma_color" ]  = std::string( "orange" );
    pack[ "residuals_windows_name" ] = std::string( "residuals" );

    pack[ "weights_data" ]         = weights;
    pack[ "weights_number_bins" ]  = uint32_t( 50 );
    pack[ "weights_windows_name" ] = std::string( "weights" );
    pack[ "weights_color" ]        = std::string( "seagreen" );

    // Mat (int rows, int cols, int type, void *data, size_t step=AUTO_STEP)
    cv::Mat cvHessianGray( m_hessian.rows(), m_hessian.cols(), CV_64F, m_hessian.data() );
    // std::cout << "cvHessianGray: "
    //   << "type: " << cvHessianGray.type() << ", rows: " << cvHessianGray.rows << ", cols: " << cvHessianGray.cols << std::endl;

    // std::cout << "Eigen Hessian: " << m_hessian << std::endl;
    // std::cout << "Opencv Hessian: " << cvHessianGray << std::endl;
    // cvHessianGray.convertTo(cvHessianGray, CV_32F);
    cv::normalize( cvHessianGray, cvHessianGray, -1, 1, cv::NORM_MINMAX, CV_32F );

    // std::cout << "cvHessianGray: "
    //   << "type: " << cvHessianGray.type() << ", rows: " << cvHessianGray.rows << ", cols: " << cvHessianGray.cols << std::endl;
    // std::cout << "Opencv Hessian: " << cvHessianGray << std::endl;
    pack[ "hessian_cv" ]           = cvHessianGray;
    pack[ "hessian_windows_name" ] = std::string( "hessian" );
    pack[ "hessian_colormap" ]     = std::string( "coolwarm" );
    // pack["hessian_colormap"] = std::string("PRGn");

    // cv::normalize( cvHessianGray, cvHessianGray, 0, 255, cv::NORM_MINMAX, CV_8U );
    // std::cout << "Opencv Hessian: " << cvHessianGray << std::endl;

    // cv::Mat cvHessianColor;
    // cv::applyColorMap( cvHessianGray, cvHessianColor, cv::COLORMAP_VIRIDIS );
    // cv::cvtColor(cvHessianColor, cvHessianColor, cv::COLOR_BGR2RGB);
    // std::cout << "cvHessianColor: "
    //   << "type: " << cvHessianColor.type() << ", rows: " << cvHessianColor.rows << ", cols: " << cvHessianColor.cols << std::endl;

    cv::Mat resPatches             = visualization::residualsPatches( m_residuals, 119, 5, 5, 5, 12 );
    pack[ "patches_cv" ]           = resPatches;
    pack[ "patches_windows_name" ] = std::string( "absolute residuals patches" );
    pack[ "patches_colormap" ]     = std::string( "cividis" );
    // pack["patches_colormap"] = std::string("gray");

    // std::cout << "cvHessianColor: "
    //   << "type: " << resPatches.type() << ", rows: " << resPatches.rows << ", cols: " << resPatches.cols << std::endl;
    // cv::cvtColor(resPatches, resPatches, cv::COLOR_BGR2RGB);

    // std::vector< std::vector< double > > data{residuals, weights};
    // std::vector< std::string > colors{"b", "g"};
    // std::vector< std::string > windowNames{"residuals", "weights", "patches", "jacobian"};
    visualization::drawHistogram( pack );
}