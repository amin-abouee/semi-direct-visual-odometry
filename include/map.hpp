#ifndef __MAP_HPP__
#define __MAP_HPP__

#include "feature.hpp"
#include "frame.hpp"
#include "point.hpp"

#include <iostream>
#include <memory>

class Map final
{
    using keyframeDistance = std::pair< const std::shared_ptr< Frame >&, double >;

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    std::vector< std::shared_ptr< Frame > > m_keyFrames;
    // std::vector< std::shared_ptr< Point > > m_trashPoints;

    explicit Map();
    Map( const Map& rhs ) = delete;
    Map( Map&& rhs )      = delete;
    Map& operator=( const Map& rhs ) = delete;
    Map& operator=( Map&& rhs ) = delete;
    ~Map()                      = default;

    // Reset the map. Delete all keyframes and reset the frame and point counters.
    void reset();

    // delete frame
    void removeFrame( std::shared_ptr< Frame >& frame );

    // delete point
    void removePoint( std::shared_ptr< Point >& point );

    // delete feature
    void removeFeature( std::shared_ptr< Feature >& feature );

    void addKeyframe( std::shared_ptr< Frame >& frame );

    std::shared_ptr< Frame >& getCloseKeyframe( std::shared_ptr< Frame >& frame ) const;

    void getCloseKeyframes( const std::shared_ptr< Frame >& frame, std::vector< keyframeDistance >& closeKeyframes ) const;

    /// Transform the whole map with rotation R, translation t and scale s.
    void transform( const Eigen::Matrix3d& R, const Eigen::Vector3d& t, const double& s ) const;

    /// Empty trash bin of deleted keyframes and map points. We don't delete the
    /// points immediately to ensure proper cleanup and to provide the visualizer
    /// a list of objects which must be removed.
    // void emptyTrash();

    std::shared_ptr< Frame >& getFurthestKeyframe( const Eigen::Vector3d& pos );

    bool getFrameById( const uint64_t id, const std::shared_ptr< Frame >& frame ) const;

    /// Return the number of keyframes in the map
    std::size_t sizeKeyframes() const;

private:
};

#endif /* __MAP_HPP__ */