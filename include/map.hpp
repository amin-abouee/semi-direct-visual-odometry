#ifndef __MAP_HPP__
#define __MAP_HPP__

#include "feature.hpp"
#include "feature_alignment.hpp"
#include "frame.hpp"
#include "point.hpp"

#include <iostream>
#include <memory>

class Map final
{
    using KeyframeDistance = std::pair< const std::shared_ptr< Frame >, double >;
    using FrameSize        = std::pair< const std::shared_ptr< Frame >, int32_t >;

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
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

    void getClosestKeyframe( std::shared_ptr< Frame >& frame, std::shared_ptr< Frame >& closestKeyframe ) const;

    void getCloseKeyframes( const std::shared_ptr< Frame >& frame, std::vector< KeyframeDistance >& closeKeyframes ) const;

    /// Transform the whole map with rotation R, translation t and scale s.
    void transform( const Eigen::Matrix3d& R, const Eigen::Vector3d& t, const double& s ) const;

    /// Empty trash bin of deleted keyframes and map points. We don't delete the
    /// points immediately to ensure proper cleanup and to provide the visualizer
    /// a list of objects which must be removed.
    // void emptyTrash();

    void getFurthestKeyframe( const Eigen::Vector3d& pos, std::shared_ptr< Frame >& furthestKeyframe ) const;

    bool getFrameById( const uint64_t id, std::shared_ptr< Frame >& lookingFrame ) const;

    /// Return the number of keyframes in the map
    std::size_t sizeKeyframes() const;

    /// Project points from the map into the image. First finds keyframes with
    /// overlapping field of view and projects only those map-points.
    void reprojectMap( const std::shared_ptr< Frame >& refFrame, std::shared_ptr< Frame >& curFrame, std::vector< FrameSize >& overlapKeyFrames );

    void addNewCandidate( const std::shared_ptr< Feature >& feature, const std::shared_ptr< Point >& point, const std::shared_ptr< Frame >& visitedFrame );

    void addCandidateToFrame( std::shared_ptr< Frame >& frame );

    void addCandidateToAllActiveKeyframes();

    void removeFrameCandidate( std::shared_ptr< Frame >& frame );

    void resetCandidates();

    /// A candidate is a point that projects into the image plane and for which we
    /// will search a maching feature in the image.
    struct Candidate
    {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        std::shared_ptr< Feature > m_feature;  //!< Feature in reference frame
        std::shared_ptr< Point > m_point;      //!< projected 2D pixel location.
        std::shared_ptr< Frame > m_visitedFrame;
        Candidate( const std::shared_ptr< Feature >& feature,
                   const std::shared_ptr< Point >& point,
                   const std::shared_ptr< Frame >& visitedFrame )
            : m_feature( feature ), m_point( point ), m_visitedFrame( visitedFrame )
        {
        }
    };

    using CandidateList = std::vector< Candidate >;
    using CandidateGrid = std::vector< std::shared_ptr< CandidateList > >;

    uint32_t m_matches;
    uint32_t m_trials;
    std::vector< std::shared_ptr< Frame > > m_keyFrames;
    CandidateList m_candidates;

private:
    /// The grid stores a set of candidate matches. For every grid cell we try to find one match.
    struct Grid
    {
        CandidateGrid m_cells;
        std::vector< int32_t > m_cellOrders;
        uint32_t m_cellSize;
        uint32_t m_gridCols;
        uint32_t m_gridRows;
    };

    // to projects all point into the new frame and create a list of features
    Grid m_grid;

    // To get the new candidate points from depth filter
    // CandidateList m_candidates;
    std::mutex m_mutexCandidates;

    bool pointQualityComparator( Candidate& lhs, Candidate& rhs );
    void initializeGrid( const std::shared_ptr< PinholeCamera >& camera, const uint32_t cellSize );
    void resetGrid();
    bool reprojectCell( std::shared_ptr< CandidateList >& candidates, std::shared_ptr< Frame >& frame );
    bool reprojectPoint( const std::shared_ptr< Frame >& frame, const std::shared_ptr< Feature >& feature );
};

#endif /* __MAP_HPP__ */