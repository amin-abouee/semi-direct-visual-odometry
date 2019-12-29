#ifndef __ALGORITHM_HPP__
#define __ALGORITHM_HPP__

#include <iostream>

#include <Eigen/Core>

#include "frame.hpp"

namespace algorithm
{
// void pointsRefCamera( const Frame& refFrame, const Frame& curFrame, Eigen::MatrixXd& pointsRefCamera );

// void pointsCurCamera( const Frame& refFrame, const Frame& curFrame, Eigen::MatrixXd& pointsCurCamera );

void points3DWorld( const Frame& refFrame, const Frame& curFrame, Eigen::MatrixXd& pointsWorld );

void normalizedDepthRefCamera( const Frame& refFrame,
                               const Eigen::MatrixXd& pointsWorld,
                               Eigen::VectorXd& normalizedDepthRefCamera );

void depthRefCamera( const Frame& refFrame, const Eigen::MatrixXd& pointsWorld, Eigen::VectorXd& depthRefCamera );

void normalizedDepthsCurCamera( const Frame& curFrame,
                                const Eigen::MatrixXd& pointsWorld,
                                Eigen::VectorXd& normalizedDepthCurCamera );

void depthsCurCamera( const Frame& curFrame, const Eigen::MatrixXd& pointsWorld, Eigen::VectorXd& depthCurCamera );

void triangulatePointHomogenousDLT( const Frame& refFrame,
                                    const Frame& curFrame,
                                    const Eigen::Vector2d& refFeature,
                                    const Eigen::Vector2d& curFeature,
                                    Eigen::Vector3d& point );

// Guide to 3D Vision Computation, Procedure 4.1, Triangulation with known camera matrices, Eq. 4.3, Page 61
void triangulatePointDLT( const Frame& refFrame,
                          const Frame& curFrame,
                          const Eigen::Vector2d& refFeature,
                          const Eigen::Vector2d& curFeature,
                          Eigen::Vector3d& point );

// https://paperpile.com/app/p/5bafd339-43e6-0f8e-b976-951e527f7a45
// Multi view geometry, Result 9.19, page 259
void decomposeEssentialMatrix( const Eigen::Matrix3d& E, Eigen::Matrix3d& R1, Eigen::Matrix3d& R2, Eigen::Vector3d& t );

void recoverPose(
  const Eigen::Matrix3d& E, const Frame& refFrame, Frame& curFrame, Eigen::Matrix3d& R, Eigen::Vector3d& t );

bool checkCheirality();

Eigen::Matrix3d hat( const Eigen::Vector3d& vec );

double computeMedian( const Eigen::VectorXd& vec );

// double computeMedianInplace( const Eigen::VectorXd& vec );

}  // namespace algorithm
#endif /* __ALGORITHM_HPP__ */