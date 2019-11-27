#ifndef __ALGORITHM_HPP__
#define __ALGORITHM_HPP__

#include <iostream>

#include <Eigen/Core>

#include "frame.hpp"

namespace Algorithm
{
void triangulatePointHomogenousDLT( const Frame& refFrame,
                                    const Frame& curFrame,
                                    Eigen::Vector2d& refFeature,
                                    Eigen::Vector2d& curFeature,
                                    Eigen::Vector3d& point );

// Guide to 3D Vision Computation, Procedure 4.1, Triangulation with known camera matrices, Eq. 4.3, Page 61
void triangulatePointDLT( const Frame& refFrame,
                          const Frame& curFrame,
                          Eigen::Vector2d& refFeature,
                          Eigen::Vector2d& curFeature,
                          Eigen::Vector3d& point );

// https://paperpile.com/app/p/5bafd339-43e6-0f8e-b976-951e527f7a45
// Multi view geometry, Result 9.19, page 259
void decomposeEssentialMatrix( Eigen::Matrix3d& E, Eigen::Matrix3d& R1, Eigen::Matrix3d& R2, Eigen::Vector3d& t );

};     // namespace Algorithm
#endif /* __ALGORITHM_HPP__ */