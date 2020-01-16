#include "nlls.hpp"
#include "algorithm.hpp"
#include "visualization.hpp"
#include "utils.hpp"

#include <any>
#include <opencv2/core/eigen.hpp>

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
    // std::cout << "Size hessian: " << m_hessian.rows() << " , " << m_hessian.cols() << std::endl;
    // std::cout << "Size gradient: " << m_gradient.size() << std::endl;
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

    bool computeJacobian = lambdaJacobianFunctor == nullptr ? false : true;
    // resetParameters(computeJacobian);

    unsigned int curIteration = 0;
    bool stop                 = false;

    double chiSquaredError = 0.0;
    // double squaredError    = 0.0;

    double stepSize = 0.0;
    // double normInfDiffPose					= 0.0;
    double normDiffPose              = 0.0;
    uint32_t cntTotalProjectedPixels = 0;
    

    double preChiSquaredError = std::numeric_limits<double>::max();
    Sophus::SE3d prePose = pose;

    while ( curIteration < m_maxIteration && !stop )
    {
        // std::cout << "pose: " << pose.params().format(utils::eigenFormat()) << std::endl;
        resetParameters(computeJacobian);
        cntTotalProjectedPixels = lambdaResidualFunctor( pose );
        tukeyWeighting( cntTotalProjectedPixels );
        // const uint32_t validpatches = std::count( curVisibility.begin(), curVisibility.end(), true );
        // std::cout << "projected points: " << cntTotalProjectedPixels << std::endl;
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
                chiSquaredError += m_residuals( i ) * m_residuals( i ) * m_weights( i );
                // squaredError += m_residuals( i ) * m_residuals( i );
            }
        }
        m_dx.noalias() = m_hessian.ldlt().solve( - m_gradient );
        // m_dx.noalias() = m_hessian.ldlt().solve( m_gradient );
        // std::cout << "With -gradient: " << (m_hessian.ldlt().solve( -m_gradient )).transpose() << std::endl;
        // std::cout << "With -dx: " << -m_dx.transpose() << std::endl;

        if ( m_dx.maxCoeff() > m_maxCoffDx || std::isnan( m_dx.cwiseAbs().minCoeff() ) )
            break;

        if (chiSquaredError > preChiSquaredError)
        {
            pose = prePose; // rollback to previous pose
            break;
        }

        prePose = pose;
        preChiSquaredError = chiSquaredError;

        stepSize     = m_dx.transpose() * m_dx;
        normDiffPose = ( m_dx ).norm() / ( pose.log() ).norm();
        if ( stepSize < m_stepSize || normDiffPose < m_normInfDiff || chiSquaredError < m_minChiSquaredError )
        {
            stop = true;
            lambdaUpdateFunctor( pose, m_dx );
            break;
        }
        else
        {
            lambdaUpdateFunctor( pose, m_dx );
        }
        visualize(cntTotalProjectedPixels);
        std::cout << "chi error: " << chiSquaredError << std::endl;
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

}

void NLLS::resetParameters (bool clearJacobian)
{
    m_hessian.setZero();
    m_gradient.setZero();
    m_residuals.setConstant( std::numeric_limits< double >::max() );
    m_weights.setConstant( 0.0 );
    m_visiblePoints.setConstant( false );
    if ( clearJacobian == true )
        m_jacobian.setZero();
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

void NLLS::visualize(const uint32_t numValidProjectedPoints)
{
    std::map<std::string, std::any> pack;

    std::vector< double > weights;
    std::vector< double > residuals;
    double sum = 0.0;
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

    // std::cout << "sum: " << sum << std::endl;

    double median = algorithm::computeMedian(m_residuals, numValidProjectedPoints);
    double mad = algorithm::computeMAD(m_residuals, numValidProjectedPoints);
    double sigma = algorithm::computeSigma(m_residuals, numValidProjectedPoints);
    pack["residuals_data"] = residuals;
    pack["residuals_color"] = std::string("slategray");
    pack["residuals_number_bins"] = uint32_t(50);
    pack["residuals_median"] = median;
    pack["residuals_median_color"] = std::string("royalblue");
    pack["residuals_mad"] = mad;
    pack["residuals_mad_color"] = std::string("gold");
    pack["residuals_sigma"] = sigma;
    pack["residuals_sigma_color"] = std::string("orange");
    pack["residuals_windows_name"] = std::string("residuals");


    pack["weights_data"] = weights;
    pack["weights_number_bins"] = uint32_t(50);
    pack["weights_windows_name"] = std::string("weights");
    pack["weights_color"] = std::string("seagreen");


    // Mat (int rows, int cols, int type, void *data, size_t step=AUTO_STEP)
    cv::Mat cvHessianGray(m_hessian.rows(), m_hessian.cols(), CV_64F, m_hessian.data());
    // std::cout << "cvHessianGray: "
            //   << "type: " << cvHessianGray.type() << ", rows: " << cvHessianGray.rows << ", cols: " << cvHessianGray.cols << std::endl;
    
    // std::cout << "Eigen Hessian: " << m_hessian << std::endl;
    // std::cout << "Opencv Hessian: " << cvHessianGray << std::endl;
    // cvHessianGray.convertTo(cvHessianGray, CV_32F);
    cv::normalize( cvHessianGray, cvHessianGray, -1, 1, cv::NORM_MINMAX, CV_32F );

    // std::cout << "cvHessianGray: "
            //   << "type: " << cvHessianGray.type() << ", rows: " << cvHessianGray.rows << ", cols: " << cvHessianGray.cols << std::endl;
    // std::cout << "Opencv Hessian: " << cvHessianGray << std::endl;
    pack["hessian_cv"] = cvHessianGray;
    pack["hessian_windows_name"] = std::string("hessian");
    pack["hessian_colormap"] = std::string("coolwarm");


    // cv::normalize( cvHessianGray, cvHessianGray, 0, 255, cv::NORM_MINMAX, CV_8U );
    // std::cout << "Opencv Hessian: " << cvHessianGray << std::endl;

    // cv::Mat cvHessianColor;
    // cv::applyColorMap( cvHessianGray, cvHessianColor, cv::COLORMAP_VIRIDIS );
    // cv::cvtColor(cvHessianColor, cvHessianColor, cv::COLOR_BGR2RGB);
    // std::cout << "cvHessianColor: "
            //   << "type: " << cvHessianColor.type() << ", rows: " << cvHessianColor.rows << ", cols: " << cvHessianColor.cols << std::endl;

    cv::Mat resPatches = visualization::residualsPatches( m_residuals, 119, 5, 5, 5, 12 );
    pack["patches_cv"] = resPatches;
    pack["patches_windows_name"] = std::string("patches");
    pack["patches_colormap"] = std::string("cividis");

    // std::cout << "cvHessianColor: "
            //   << "type: " << resPatches.type() << ", rows: " << resPatches.rows << ", cols: " << resPatches.cols << std::endl;
    // cv::cvtColor(resPatches, resPatches, cv::COLOR_BGR2RGB);


    // std::vector< std::vector< double > > data{residuals, weights};
    // std::vector< std::string > colors{"b", "g"};
    // std::vector< std::string > windowNames{"residuals", "weights", "patches", "jacobian"};
    visualization::drawHistogram( pack );
}