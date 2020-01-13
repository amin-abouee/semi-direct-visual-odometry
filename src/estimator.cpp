#include "estimator.hpp"

/**
 * Estimator implementation
 * Reference: Parameter Estimation Techniques: A Tutorial with Application to Conic Fitting
 */

std::map< std::string, Estimator::EstimatorModel > Estimator::allMethods{
  std::make_pair( "l2", Estimator::EstimatorModel::L2 ),
  std::make_pair( "l1", Estimator::EstimatorModel::L1 ),
  std::make_pair( "diff", Estimator::EstimatorModel::Diff ),
  std::make_pair( "lp", Estimator::EstimatorModel::Lp ),
  std::make_pair( "fair", Estimator::EstimatorModel::Fair ),
  std::make_pair( "huber", Estimator::EstimatorModel::Huber ),
  std::make_pair( "cauchy", Estimator::EstimatorModel::Cauchy ),
  std::make_pair( "geman-mcclure", Estimator::EstimatorModel::GemanMcClure ),
  std::make_pair( "welch", Estimator::EstimatorModel::Welch ),
  std::make_pair( "tukey", Estimator::EstimatorModel::Tukey ),
  std::make_pair( "drummond", Estimator::EstimatorModel::Drummond ),
  std::make_pair( "andrew-wave", Estimator::EstimatorModel::AndrewWave ),
  std::make_pair( "ramsay", Estimator::EstimatorModel::Ramsay ),
  std::make_pair( "trimmed-mean", Estimator::EstimatorModel::TrimmedMean ),
  std::make_pair( "t-distro", Estimator::EstimatorModel::TDistro )};

void Estimator::MEstimator( const Eigen::Ref< Eigen::VectorXd >& residuals,
                            Eigen::Ref< Eigen::VectorXd >& weightVector,
                            const EstimatorModel& model )
{
    // std::cout << "in estimator add: " << &residuals(0) << std::endl;
    switch ( model )
    {
    case EstimatorModel::L2:
        // std::cout << "Mode: L2" << std::endl;
        computeL2( residuals, weightVector );
        break;
    case EstimatorModel::L1:
        // std::cout << "Mode: L1" << std::endl;
        computeL1( residuals, weightVector );
        break;
    case EstimatorModel::Diff:
        // std::cout << "Mode: Diff" << std::endl;
        computeL1L2( residuals, weightVector );
        break;
    case EstimatorModel::Lp:
        // std::cout << "Mode: Lp" << std::endl;
        computeLp( residuals, weightVector );
        break;
    case EstimatorModel::Fair:
        // std::cout << "Mode: Fair" << std::endl;
        computeFair( residuals, weightVector );
        break;
    case EstimatorModel::Huber:
        // std::cout << "Mode: Huber" << std::endl;
        computeHuber( residuals, weightVector );
        break;
    case EstimatorModel::Cauchy:
        // std::cout << "Mode: Cauchy" << std::endl;
        computeCauchy( residuals, weightVector );
        break;
    case EstimatorModel::GemanMcClure:
        // std::cout << "Mode: GemanMcClure" << std::endl;
        computeGemanMcClure( residuals, weightVector );
        break;
    case EstimatorModel::Welch:
        // std::cout << "Mode: Welch" << std::endl;
        computeWelch( residuals, weightVector );
        break;
    case EstimatorModel::Tukey:
        // std::cout << "Mode: Tukey" << std::endl;
        computeTukey( residuals, weightVector );
        break;
    case EstimatorModel::Drummond:
        // std::cout << "Mode: Drummond" << std::endl;
        computeDrummond( residuals, weightVector );
        break;
    case EstimatorModel::AndrewWave:
        // std::cout << "Mode: AndrewWave" << std::endl;
        computeAndrewWave( residuals, weightVector );
        break;
    case EstimatorModel::Ramsay:
        // std::cout << "Mode: Ramsay" << std::endl;
        computeRamsay( residuals, weightVector );
        break;
    case EstimatorModel::TrimmedMean:
        // std::cout << "Mode: TrimmedMean" << std::endl;
        computeTrimmedMean( residuals, weightVector );
        break;
    case EstimatorModel::TDistro:
        // std::cout << "Mode: TrimmedMean" << std::endl;
        computeTDistro( residuals, weightVector );
        break;
    }
}

void Estimator::MEstimator( Eigen::Map< Eigen::VectorXd >& residuals,
                            Eigen::Map< Eigen::VectorXd >& weightVector,
                            const EstimatorModel& model )
{
	// checked, doesn't allocate memory
    Eigen::Ref< Eigen::VectorXd > res    = residuals.head( residuals.rows() );
	// std::cout << "addr residual: " << &residuals(0) << std::endl;
	// std::cout << "addr res: " << &res(0) << std::endl;
    Eigen::Ref< Eigen::VectorXd > weight = weightVector.head( weightVector.rows() );
    MEstimator( res, weight, model );
}

double Estimator::computeSTD( const Eigen::Ref< Eigen::VectorXd >& residuals )
{
    const std::size_t numObservations                          = residuals.rows();
    Eigen::Matrix< double, Eigen::Dynamic, 1 > sortedResiduals = residuals.cwiseAbs();
    std::sort( sortedResiduals.data(), sortedResiduals.data() + numObservations );
    double median = 0.0;
    if ( numObservations % 2 == 0 )
        median = ( sortedResiduals( numObservations / 2 - 1 ) + sortedResiduals( numObservations / 2 ) ) / 2;
    else
        median = sortedResiduals( numObservations / 2 );

    return 1.4826 * ( 1 + ( 5 / ( numObservations - 6 ) ) ) * median;
}

void Estimator::computeL2( const Eigen::Ref< Eigen::VectorXd >& residuals, Eigen::Ref< Eigen::VectorXd >& weightVector )
{
    const std::size_t numObservations = residuals.rows();
    for ( std::size_t i( 0 ); i < numObservations; i++ )
    {
        weightVector( i ) = 1.0;
    }
}

void Estimator::computeL1( const Eigen::Ref< Eigen::VectorXd >& residuals, Eigen::Ref< Eigen::VectorXd >& weightVector )
{
    const std::size_t numObservations = residuals.rows();
    for ( std::size_t i( 0 ); i < numObservations; i++ )
    {
        weightVector( i ) = 1.0 / std::abs( residuals( i ) );
    }
}

void Estimator::computeL1L2( const Eigen::Ref< Eigen::VectorXd >& residuals,
                             Eigen::Ref< Eigen::VectorXd >& weightVector )
{
    const std::size_t numObservations = residuals.rows();
    for ( std::size_t i( 0 ); i < numObservations; i++ )
    {
        weightVector( i ) = 1.0 / std::sqrt( 1 + ( residuals( i ) * residuals( i ) / 2 ) );
    }
}

void Estimator::computeLp( const Eigen::Ref< Eigen::VectorXd >& residuals, Eigen::Ref< Eigen::VectorXd >& weightVector )
{
    const std::size_t numObservations = residuals.rows();
    for ( std::size_t i( 0 ); i < numObservations; i++ )
    {
        weightVector( i ) = 1.0 / std::pow( std::abs( residuals( i ) ), 1.2 );
    }
}

void Estimator::computeFair( const Eigen::Ref< Eigen::VectorXd >& residuals,
                             Eigen::Ref< Eigen::VectorXd >& weightVector )
{
    double sigma = computeSTD( residuals );
    if ( sigma <= std::numeric_limits< double >::epsilon() )
        sigma = std::numeric_limits< double >::epsilon();
    const double c                    = 1.3998 * sigma;
    const std::size_t numObservations = residuals.rows();
    for ( std::size_t i( 0 ); i < numObservations; i++ )
    {
        weightVector( i ) = 1.0 / ( 1 + ( std::abs( residuals( i ) ) / c ) );
    }
}

void Estimator::computeHuber( const Eigen::Ref< Eigen::VectorXd >& residuals,
                              Eigen::Ref< Eigen::VectorXd >& weightVector )
{
    const double sigma                = computeSTD( residuals );
    const double c                    = 1.345 * sigma;
    const std::size_t numObservations = residuals.rows();
    for ( std::size_t i( 0 ); i < numObservations; i++ )
    {
        double abs = std::abs( residuals( i ) );
        if ( abs <= c )
            weightVector( i ) = 1.0;
        else
            weightVector( i ) = c / abs;
    }
}

void Estimator::computeCauchy( const Eigen::Ref< Eigen::VectorXd >& residuals,
                               Eigen::Ref< Eigen::VectorXd >& weightVector )
{
    double sigma = computeSTD( residuals );
    if ( sigma <= std::numeric_limits< double >::epsilon() )
        sigma = std::numeric_limits< double >::epsilon();
    const double c                    = 2.3849 * sigma;
    const std::size_t numObservations = residuals.rows();
    for ( std::size_t i( 0 ); i < numObservations; i++ )
    {
        weightVector( i ) = 1.0 / ( 1 + ( ( residuals( i ) * residuals( i ) ) / ( c * c ) ) );
    }
}

void Estimator::computeGemanMcClure( const Eigen::Ref< Eigen::VectorXd >& residuals,
                                     Eigen::Ref< Eigen::VectorXd >& weightVector )
{
    const std::size_t numObservations = residuals.rows();
    for ( std::size_t i( 0 ); i < numObservations; i++ )
    {
        weightVector( i ) = 1.0 / std::pow( 1 + residuals( i ) * residuals( i ), 2 );
    }
}

void Estimator::computeWelch( const Eigen::Ref< Eigen::VectorXd >& residuals,
                              Eigen::Ref< Eigen::VectorXd >& weightVector )
{
    double sigma = computeSTD( residuals );
    if ( sigma <= std::numeric_limits< double >::epsilon() )
        sigma = std::numeric_limits< double >::epsilon();
    const double c                    = 2.9846 * sigma;
    const std::size_t numObservations = residuals.rows();
    for ( std::size_t i( 0 ); i < numObservations; i++ )
    {
        weightVector( i ) = std::exp( -( residuals( i ) * residuals( i ) ) / ( c * c ) );
    }
}

void Estimator::computeTukey( const Eigen::Ref< Eigen::VectorXd >& residuals,
                              Eigen::Ref< Eigen::VectorXd >& weightVector )
{
    double sigma = computeSTD( residuals );
    if ( sigma <= std::numeric_limits< double >::epsilon() )
        sigma = std::numeric_limits< double >::epsilon();
    const double c                    = 4.6851 * sigma;
    const std::size_t numObservations = residuals.rows();
    for ( std::size_t i( 0 ); i < numObservations; i++ )
    {
        double abs = std::abs( residuals( i ) );
        if ( abs <= c )
            weightVector( i ) = std::pow( 1.0 - ( ( residuals( i ) * residuals( i ) ) / ( c * c ) ), 2 );
        else
            weightVector( i ) = 0;
    }
    // std::cout << "bn: " << residuals/(residuals.size()+1) << std::endl;
}

void Estimator::computeDrummond( const Eigen::Ref< Eigen::VectorXd >& residuals,
                                 Eigen::Ref< Eigen::VectorXd >& weightVector )
{
    const double sigma                = computeSTD( residuals );
    const std::size_t numObservations = residuals.rows();
    for ( std::size_t i( 0 ); i < numObservations; i++ )
    {
        weightVector( i ) = 1.0 / ( std::abs( residuals( i ) + sigma ) );
    }
}

void Estimator::computeAndrewWave( const Eigen::Ref< Eigen::VectorXd >& residuals,
                                   Eigen::Ref< Eigen::VectorXd >& weightVector )
{
    double sigma = computeSTD( residuals );
    if ( sigma <= std::numeric_limits< double >::epsilon() )
        sigma = std::numeric_limits< double >::epsilon();
    const double c                    = 1.3387 * sigma;
    const std::size_t numObservations = residuals.rows();
    for ( std::size_t i( 0 ); i < numObservations; i++ )
    {
        double abs = std::abs( residuals( i ) );
        if ( abs <= c * M_PI )
            weightVector( i ) = std::sin( residuals( i ) / c ) / ( residuals( i ) / c );
        else
            weightVector( i ) = 0;
    }
}

void Estimator::computeRamsay( const Eigen::Ref< Eigen::VectorXd >& residuals,
                               Eigen::Ref< Eigen::VectorXd >& weightVector )
{
    const double sigma                = computeSTD( residuals );
    const std::size_t numObservations = residuals.rows();
    for ( std::size_t i( 0 ); i < numObservations; i++ )
    {
        weightVector( i ) = std::exp( -( residuals( i ) * sigma ) );
    }
}

void Estimator::computeTrimmedMean( const Eigen::Ref< Eigen::VectorXd >& residuals,
                                    Eigen::Ref< Eigen::VectorXd >& weightVector )
{
    const double sigma                = computeSTD( residuals );
    const std::size_t numObservations = residuals.rows();
    for ( std::size_t i( 0 ); i < numObservations; i++ )
    {
        double abs = std::abs( residuals( i ) );
        if ( abs <= sigma )
            weightVector( i ) = 1.0;
        else
            weightVector( i ) = 0.0;
    }
}

void Estimator::computeTDistro( const Eigen::Ref< Eigen::VectorXd >& residuals,
                                Eigen::Ref< Eigen::VectorXd >& weightVector )
{
    double sigma = computeSTD( residuals );
    if ( sigma <= std::numeric_limits< double >::epsilon() )
        sigma = std::numeric_limits< double >::epsilon();
    const std::size_t numObservations = residuals.rows();
    for ( std::size_t i( 0 ); i < numObservations; i++ )
    {
        weightVector( i ) = 6 / ( 5 + ( ( residuals( i ) * residuals( i ) ) / ( sigma * sigma ) ) );
    }
}

void Estimator::computeGeneralFunctionBarron( const Eigen::Ref< Eigen::VectorXd >& residuals,
                                              Eigen::Ref< Eigen::VectorXd >& weightVector,
                                              const double alpha )
{
    const double z = std::max( 1.0, 2 - alpha );
    double c       = computeSTD( residuals );
    if ( c <= std::numeric_limits< double >::epsilon() )
        c = std::numeric_limits< double >::epsilon();
    const std::size_t numObservations = residuals.rows();
    for ( std::size_t i( 0 ); i < numObservations; i++ )
    {
        if ( alpha == 0 )
            weightVector( i ) = 2 / ( residuals( i ) * residuals( i ) + 2 * c * c );
        else if ( alpha == -std::numeric_limits< double >::infinity() )
            weightVector( i ) = ( 1 / ( c * c ) ) * std::exp( -0.5 * ( residuals( i ) * residuals( i ) / ( c * c ) ) );
        else
            weightVector( i ) =
              ( 1 / ( c * c ) ) *
              std::pow( ( ( ( residuals( i ) * residuals( i ) / ( c * c ) ) / z ) + 1 ), alpha / 2 - 1 );
    }
}

void Estimator::TDistribution( const Eigen::Ref< Eigen::VectorXd >& residuals,
                               Eigen::Ref< Eigen::VectorXd >& weightVector,
                               const double error )
{
    const std::size_t numObservations = residuals.rows();
    double sigma                      = 0.0;
    if ( error == 0 )
        sigma = computeSTD( residuals );
    else
        sigma = ( 1 / numObservations ) * error;

    if ( sigma <= std::numeric_limits< double >::epsilon() )
        sigma = std::numeric_limits< double >::epsilon();

    for ( std::size_t i( 0 ); i < numObservations; i++ )
    {
        weightVector( i ) = 6 / ( 5 + ( ( residuals( i ) * residuals( i ) ) / ( sigma * sigma ) ) );
    }
}
