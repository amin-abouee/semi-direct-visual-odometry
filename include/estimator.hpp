/**
 * Project Inside Out Tracking
 * @version 0.6.0
 *
 * @file estimator
 * @brief M-estimator for tracking
 *
 * @date 23.02.2018
 * @author Amin Abouee
 *
 * @section DESCRIPTION
 *
 *
 *
 */

#ifndef _ESTIMATOR_H
#define _ESTIMATOR_H

#include <Eigen/Core>
#include <cmath>
#include <iostream>
#include <map>

#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif

class Estimator final
{
public:
	enum class EstimatorModel : unsigned int
	{
		L2			 = 0,
		L1			 = 1,
		Diff		 = 2,
		Lp			 = 3,
		Fair		 = 4,
		Huber		 = 5,
		Cauchy		 = 6,
		GemanMcClure = 7,
		Welch		 = 8,
		Tukey		 = 9,
		Drummond	 = 10,
		AndrewWave   = 11,
		Ramsay		 = 12,
		TrimmedMean  = 13,
		TDistro		 = 14
	};

	// C'tor
	explicit Estimator() = default;

	// D'tor
	virtual ~Estimator() = default;

	/**
	 * @param residuals
	 * @param model
	 */
	void MEstimator( const Eigen::Ref< Eigen::VectorXd >& residuals,
					 Eigen::Ref< Eigen::VectorXd >& weightVector,
					 const EstimatorModel& model );

	void MEstimator( Eigen::Map <Eigen::VectorXd>& residuals,
					 Eigen::Map <Eigen::VectorXd>& weightVector,
					 const EstimatorModel& model );

	void TDistribution( const Eigen::Ref< Eigen::VectorXd >& residuals,
						Eigen::Ref< Eigen::VectorXd >& weightVector,
						const double error );

	static std::map <std::string, EstimatorModel> allMethods;

private:
	double computeSTD( const Eigen::Ref< Eigen::VectorXd >& residuals );

	void computeL2( const Eigen::Ref< Eigen::VectorXd >& residuals, Eigen::Ref< Eigen::VectorXd >& weightVector );
	void computeL1( const Eigen::Ref< Eigen::VectorXd >& residuals, Eigen::Ref< Eigen::VectorXd >& weightVector );
	void computeL1L2( const Eigen::Ref< Eigen::VectorXd >& residuals, Eigen::Ref< Eigen::VectorXd >& weightVector );
	void computeLp( const Eigen::Ref< Eigen::VectorXd >& residuals, Eigen::Ref< Eigen::VectorXd >& weightVector );
	void computeFair( const Eigen::Ref< Eigen::VectorXd >& residuals, Eigen::Ref< Eigen::VectorXd >& weightVector );
	void computeHuber( const Eigen::Ref< Eigen::VectorXd >& residuals, Eigen::Ref< Eigen::VectorXd >& weightVector );
	void computeCauchy( const Eigen::Ref< Eigen::VectorXd >& residuals, Eigen::Ref< Eigen::VectorXd >& weightVector );
	void computeGemanMcClure( const Eigen::Ref< Eigen::VectorXd >& residuals,
							  Eigen::Ref< Eigen::VectorXd >& weightVector );
	void computeWelch( const Eigen::Ref< Eigen::VectorXd >& residuals, Eigen::Ref< Eigen::VectorXd >& weightVector );
	void computeTukey( const Eigen::Ref< Eigen::VectorXd >& residuals, Eigen::Ref< Eigen::VectorXd >& weightVector );
	void computeDrummond( const Eigen::Ref< Eigen::VectorXd >& residuals, Eigen::Ref< Eigen::VectorXd >& weightVector );

	// http://www.statsmodels.org/dev/examples/notebooks/generated/robust_models_1.html
	void computeAndrewWave( const Eigen::Ref< Eigen::VectorXd >& residuals, Eigen::Ref< Eigen::VectorXd >& weightVector );
	void computeRamsay( const Eigen::Ref< Eigen::VectorXd >& residuals, Eigen::Ref< Eigen::VectorXd >& weightVector );
	void computeTrimmedMean( const Eigen::Ref< Eigen::VectorXd >& residuals,
							 Eigen::Ref< Eigen::VectorXd >& weightVector );
	void computeTDistro( const Eigen::Ref< Eigen::VectorXd >& residuals, Eigen::Ref< Eigen::VectorXd >& weightVector );
	// A More General Robust Loss Function (Jonathan T. Barron)
	// https://arxiv.org/abs/1701.03077
	void computeGeneralFunctionBarron( const Eigen::Ref< Eigen::VectorXd >& residuals,
									   Eigen::Ref< Eigen::VectorXd >& weightVector,
									   const double alpha );
};

#endif  //_ESTIMATOR_H
