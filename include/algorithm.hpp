#ifndef __ALGORITHM_HPP__
#define __ALGORITHM_HPP__

#include <iostream>

#include <Eigen/Core>

#include "frame.hpp"

namespace algorithm
{
// https://forum.kde.org/viewtopic.php?f=74&t=96718
using MapXRow      = Eigen::Map< Eigen::Matrix< uint8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor > >;
using MapXRowConst = Eigen::Map< const Eigen::Matrix< uint8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor > >;
// void pointsRefCamera( const Frame& refFrame, const Frame& curFrame, Eigen::MatrixXd& pointsRefCamera );

// void pointsCurCamera( const Frame& refFrame, const Frame& curFrame, Eigen::MatrixXd& pointsCurCamera );

void points3DWorld( const Frame& refFrame, const Frame& curFrame, Eigen::MatrixXd& pointsWorld );

void transferPointsWorldToCam( const Frame& frame, const Eigen::MatrixXd& pointsWorld, Eigen::MatrixXd& pointsCamera );

void transferPointsCamToWorld( const Frame& frame, const Eigen::MatrixXd& pointsCamera, Eigen::MatrixXd& pointsWorld );

void normalizedDepthCamera( const Frame& frame, const Eigen::MatrixXd& pointsWorld, Eigen::VectorXd& normalizedDepthCamera );

void normalizedDepthCamera( const Frame& frame, Eigen::VectorXd& normalizedDepthCamera );

void depthCamera( const Frame& frame, const Eigen::MatrixXd& pointsWorld, Eigen::VectorXd& depthCamera );

void depthCamera( const Frame& frame, Eigen::VectorXd& depthCamera );

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

void recoverPose( const Eigen::Matrix3d& E, const Frame& refFrame, Frame& curFrame, Eigen::Matrix3d& R, Eigen::Vector3d& t );

Sophus::SE3d computeRelativePose( const Frame& refFrame, const Frame& curFrame );

Eigen::Matrix3d hat( const Eigen::Vector3d& vec );

double computeMedian( const Eigen::VectorXd& input );

double computeMedian( const Eigen::VectorXd& input, const uint32_t numValidPoints );

double computeMAD( const Eigen::VectorXd& input, const uint32_t numValidPoints );

double computeSigma( const Eigen::VectorXd& input, const uint32_t numValidPoints, const double k = 1.482602218505602 );

float bilinearInterpolation( const MapXRow& image, const double x, const double y );

float bilinearInterpolation( const MapXRowConst& image, const double x, const double y );

// double computeMedianInplace( const Eigen::VectorXd& vec );

}  // namespace algorithm
#endif /* __ALGORITHM_HPP__ */