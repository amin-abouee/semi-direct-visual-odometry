#ifndef __TRIANGULATION_HPP__
#define __TRIANGULATION_HPP__

#include <iostream>

#include <Eigen/Core>

#include "frame.hpp"

class Triangulation final
{
public:
    explicit Triangulation();
    Triangulation( const Triangulation& rhs ) = delete;
    Triangulation( Triangulation&& rhs )      = delete;
    Triangulation& operator=( const Triangulation& rhs ) = delete;
    Triangulation& operator=( Triangulation&& rhs ) = delete;
    ~Triangulation()                                = default;

    void triangulatePointDLT( const Frame& refFrame,
                              const Frame& curFrame,
                              Eigen::Vector2d& refFeature,
                              Eigen::Vector2d& curFeature,
                              Eigen::Vector3d& point );

private:
};
#endif /* __TRIANGULATION_HPP__ */