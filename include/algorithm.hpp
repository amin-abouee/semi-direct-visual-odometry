#ifndef __ALGORITHM_HPP__
#define __ALGORITHM_HPP__

#include <iostream>
#include <memory>

#include <Eigen/Core>

#include "frame.hpp"
#include "point.hpp"
// #define Algorithm_Log( LEVEL ) CLOG( LEVEL, "Algorithm" )

namespace algorithm
{
// https://forum.kde.org/viewtopic.php?f=74&t=96718
using MapXRow      = Eigen::Map< Eigen::Matrix< uint8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor > >;
using MapXRowConst = Eigen::Map< const Eigen::Matrix< uint8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor > >;

bool computeOpticalFlowSparse( std::shared_ptr< Frame >& refFrame,
                               std::shared_ptr< Frame >& curFrame,
                               const uint32_t patchSize,
                               const double disparityThreshold );

bool computeEssentialMatrix( std::shared_ptr< Frame >& refFrame,
                             std::shared_ptr< Frame >& curFrame,
                             const double reproError,
                             const uint32_t thresholdCorrespondingPoints,
                             Eigen::Matrix3d& E );

// https://paperpile.com/app/p/5bafd339-43e6-0f8e-b976-951e527f7a45
// Multi view geometry, Result 9.19, page 259
void decomposeEssentialMatrix( const Eigen::Matrix3d& E, Eigen::Matrix3d& R1, Eigen::Matrix3d& R2, Eigen::Vector3d& t );

bool recoverPose( const Eigen::Matrix3d& E,
                  const std::shared_ptr< Frame >& refFrame,
                  std::shared_ptr< Frame >& curFrame,
                  Eigen::Matrix3d& R,
                  Eigen::Vector3d& t );

// void templateMatching( const std::shared_ptr< Frame >& refFrame,
//                        std::shared_ptr< Frame >& curFrame,
//                        const uint16_t patchSzRef,
//                        const uint16_t patchSzCur );

void getAffineWarp( const std::shared_ptr< Frame >& refFrame,
                    const std::shared_ptr< Frame >& curFrame,
                    const std::shared_ptr< Feature >& feature,
                    const Sophus::SE3d& relativePose,
                    const uint32_t patchSize,
                    const double depth,
                    Eigen::Matrix2d& affineWarp );

void applyAffineWarp( const std::shared_ptr< Frame >& frame,
                      const Eigen::Vector2d& point,
                      const int32_t halfPatchSize,
                      const Eigen::Matrix2d& affineWarp,
                      const double boundary,
                      Eigen::Matrix< uint8_t, Eigen::Dynamic, 1 >& data );

double computeScore( const Eigen::Matrix< uint8_t, Eigen::Dynamic, 1 >& refPatchIntensity,
                     const Eigen::Matrix< uint8_t, Eigen::Dynamic, 1 >& curPatchIntensity );

// bool matchDirect( const std::shared_ptr< Point >& point, const std::shared_ptr< Frame >& curFrame, Eigen::Vector2d& feature );

bool matchEpipolarConstraint( const std::shared_ptr< Frame >& refFrame,
                              const std::shared_ptr< Frame >& curFrame,
                              std::shared_ptr< Feature >& refFeature,
                              const uint32_t patchSize,
                              const double initialDepth,
                              const double minDepth,
                              const double maxDepth,
                              double& estimatedDepth );

void triangulate3DWorldPoints( const std::shared_ptr< Frame >& refFrame,
                               const std::shared_ptr< Frame >& curFrame,
                               Eigen::MatrixXd& pointsWorld );

void transferPointsWorldToCam( const std::shared_ptr< Frame >& frame, const Eigen::MatrixXd& pointsWorld, Eigen::MatrixXd& pointsCamera );

void transferPointsCamToWorld( const std::shared_ptr< Frame >& frame, const Eigen::MatrixXd& pointsCamera, Eigen::MatrixXd& pointsWorld );

void normalizedDepthCamera( const std::shared_ptr< Frame >& frame,
                            const Eigen::MatrixXd& pointsWorld,
                            Eigen::VectorXd& normalizedDepthCamera );

void normalizedDepthCamera( const std::shared_ptr< Frame >& frame, Eigen::VectorXd& normalizedDepthCamera );

void depthCamera( const std::shared_ptr< Frame >& frame, const Eigen::MatrixXd& pointsWorld, Eigen::VectorXd& depthCamera );

void depthCamera( const std::shared_ptr< Frame >& frame, Eigen::VectorXd& depthCamera );

void triangulatePointHomogenousDLT( const std::shared_ptr< Frame >& refFrame,
                                    const std::shared_ptr< Frame >& curFrame,
                                    const Eigen::Vector2d& refFeature,
                                    const Eigen::Vector2d& curFeature,
                                    Eigen::Vector3d& point );

// Guide to 3D Vision Computation, Procedure 4.1, Triangulation with known camera matrices, Eq. 4.3, Page 61
void triangulatePointDLT( const std::shared_ptr< Frame >& refFrame,
                          const std::shared_ptr< Frame >& curFrame,
                          const Eigen::Vector2d& refFeature,
                          const Eigen::Vector2d& curFeature,
                          Eigen::Vector3d& point );

bool depthFromTriangulation( const Sophus::SE3d& relativePose,
                             const Eigen::Vector3d& refBearingVec,
                             const Eigen::Vector3d& curBearingVec,
                             double& depth );

Sophus::SE3d computeRelativePose( const std::shared_ptr< Frame >& refFrame, const std::shared_ptr< Frame >& curFrame );

double computeStructureError( const std::shared_ptr< Point >& point );

uint32_t computeNumberProjectedPoints( const std::shared_ptr< Frame >& curFrame );

Eigen::Matrix3d hat( const Eigen::Vector3d& vec );

double computeMedian( const Eigen::VectorXd& input );

double computeMedian( const Eigen::VectorXd& input, const uint32_t numValidPoints );

double computeMAD( const Eigen::VectorXd& input, const uint32_t numValidPoints );

double computeSigma( const Eigen::VectorXd& input, const uint32_t numValidPoints, const double k = 1.482602218505602 );

float bilinearInterpolation( const MapXRow& image, const double x, const double y );

float bilinearInterpolation( const MapXRowConst& image, const double x, const double y );

double bilinearInterpolationDouble( const MapXRowConst& image, const double x, const double y );

double computeNormalDistribution( const double mu, const double sigma, const double x );

// double computeMedianInplace( const Eigen::VectorXd& vec );

}  // namespace algorithm
#endif /* __ALGORITHM_HPP__ */