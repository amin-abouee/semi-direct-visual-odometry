#ifndef __NLLS_HPP__
#define __NLLS_HPP__

#include <cmath>
#include <iostream>

#include <Eigen/Core>
#include <sophus/se3.hpp>

#include "estimator.hpp"

class NLLS
{
public:
    enum class LevenbergMethod : unsigned int
    {
        Marquardt = 0,
        Quadratic = 1,
        Nielsen   = 2
    };

    uint32_t m_numUnknowns;
    Eigen::MatrixXd m_hessian;
    Eigen::MatrixXd m_jacobian;
    Eigen::VectorXd m_residuals;
    Eigen::VectorXd m_gradient;
    Eigen::VectorXd m_weights;
    Eigen::VectorXd m_dx;
    Eigen::Matrix<bool, Eigen::Dynamic, 1> m_visiblePoints;

    // https://stackoverflow.com/a/26904458/1804533

    LevenbergMethod m_levenbergMethod;
    uint32_t m_maxIteration;
    double m_stepSize;
    double m_minChiSquaredError;
    double m_normInfDiff;
    double m_maxCoffDx;
    Estimator::EstimatorModel m_estimatorModel;
    bool m_enableErrorAnalysis;

    double m_qualityFit;
    Eigen::Matrix< double, 6, 6 > m_covarianceMatrixParameters;
    Eigen::Matrix< double, 6, 1 > m_asymptoticStandardUncertaintyParameter;

    explicit NLLS( const uint32_t numUnknowns );
    NLLS( const NLLS& rhs );
    NLLS( NLLS&& rhs );
    NLLS& operator=( const NLLS& rhs );
    NLLS& operator=( NLLS&& rhs );
    ~NLLS()       = default;

    // double optimizeGN( Sophus::SE3d& pose,
    //             const std::function< unsigned int( Sophus::SE3d& pose) >& lambdaResidualFunctor,
    //             const std::size_t numObservations);

    // double optimizeGN( Sophus::SE3d& pose,
    //             const std::function< unsigned int( Sophus::SE3d& pose) >& lambdaResidualFunctor,
    //             const std::function< unsigned int( Sophus::SE3d& pose) >& lambdaJacobianFunctor,
    //             const std::size_t numObservations);

    double optimizeGN( Sophus::SE3d& pose,
                       const std::function< uint32_t( Sophus::SE3d& pose ) >& lambdaResidualFunctor,
                       const std::function< uint32_t( Sophus::SE3d& pose ) >& lambdaJacobianFunctor,
                       const std::function< void( Sophus::SE3d& pose, const Eigen::VectorXd& dx ) >& lambdaUpdateFunctor,
                       const std::size_t numObservations);

    double optimizeLM( Sophus::SE3d& pose,
                       const std::function< uint32_t( Sophus::SE3d&, Eigen::VectorXd& res ) >& lambdaResidualFunctor,
                       const std::function< uint32_t( Sophus::SE3d&, Eigen::MatrixXd& jac ) >& lambdaJacobianFunctor,
                       const std::function< void( Sophus::SE3d& pose, const Eigen::VectorXd& dx ) >& lambdaUpdateFunctor,
                       const std::size_t numObservations );

    void initParameters(const std::size_t numObservations);

private:

    void resetParameters (bool clearJacobian=false);

    void tukeyWeighting (const uint32_t numValidProjectedPoints);

    void visualize(const uint32_t numValidProjectedPoints);
};

#endif /* __NLLS_HPP__ */
