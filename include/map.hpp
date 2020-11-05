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
    using frameSize        = std::pair< const std::shared_ptr< Frame >&, int32_t>;

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    std::vector< std::shared_ptr< Frame > > m_keyFrames;
    // std::vector< std::shared_ptr< Point > > m_trashPoints;

    explicit Map( const std::shared_ptr< PinholeCamera >& camera, const uint32_t cellSize );
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

    std::shared_ptr< Frame >& getClosestKeyframe( std::shared_ptr< Frame >& frame ) const;

    void getCloseKeyframes( const std::shared_ptr< Frame >& frame, std::vector< keyframeDistance >& closeKeyframes ) const;

    /// Transform the whole map with rotation R, translation t and scale s.
    void transform( const Eigen::Matrix3d& R, const Eigen::Vector3d& t, const double& s ) const;

    /// Empty trash bin of deleted keyframes and map points. We don't delete the
    /// points immediately to ensure proper cleanup and to provide the visualizer
    /// a list of objects which must be removed.
    // void emptyTrash();

    std::shared_ptr< Frame >& getFurthestKeyframe( const Eigen::Vector3d& pos );

    bool getFrameById( const uint64_t id, std::shared_ptr< Frame >& lookingFrame ) const;

    /// Return the number of keyframes in the map
    std::size_t sizeKeyframes() const;

    /// Project points from the map into the image. First finds keyframes with
    /// overlapping field of view and projects only those map-points.
    void reprojectMap( std::shared_ptr< Frame >& frame,
                       std::vector< frameSize >& overlapKeyFrames );

    int32_t m_matches;
    int32_t m_trials;

private:
    /// A candidate is a point that projects into the image plane and for which we
    /// will search a maching feature in the image.
    struct Candidate
    {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        std::shared_ptr< Point > m_point;  //!< 3D point.
        Eigen::Vector2d m_feature;         //!< projected 2D pixel location.
        Candidate( const std::shared_ptr< Point >& point, const Eigen::Vector2d& feature ) : m_point( point ), m_feature( feature )
        {
        }
    };

    using Cell          = std::vector< Candidate >;
    using CandidateGrid = std::vector< std::shared_ptr< Cell > >;

    /// The grid stores a set of candidate matches. For every grid cell we try to find one match.
    struct Grid
    {
        CandidateGrid m_cells;
        std::vector< int32_t > m_cellOrders;
        uint32_t m_cellSize;
        uint32_t m_gridCols;
        uint32_t m_gridRows;
    };

    Grid m_grid;

    bool pointQualityComparator( Candidate& lhs, Candidate& rhs );
    void initializeGrid( const std::shared_ptr< PinholeCamera >& camera, const uint32_t cellSize );
    void resetGrid();
    bool reprojectCell( std::shared_ptr<Cell>& cell, std::shared_ptr< Frame >& frame );
    bool reprojectPoint( const std::shared_ptr< Frame >& frame, const std::shared_ptr< Point >& point );
};

#endif /* __MAP_HPP__ */