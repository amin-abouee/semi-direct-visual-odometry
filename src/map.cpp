#include "map.hpp"
#include "algorithm.hpp"

#include <algorithm>
#include <memory>
#include <numeric>
#include <random>
#include <utility>


#include "easylogging++.h"
#define Map_Log( LEVEL ) CLOG( LEVEL, "Map" )

Map::Map( const std::shared_ptr< PinholeCamera >& camera, const uint32_t cellSize )
{
    initializeGrid( camera, cellSize );
}

void Map::reset()
{
    m_keyFrames.clear();
    // m_trashPoints.clear();
    // emptyTrash();
}

void Map::removeFrame( std::shared_ptr< Frame >& frame )
{
    // auto find = [ &frame ]( std::shared_ptr< Frame >& f ) -> bool {
    //     // TODO: check with get() and raw pointer
    //     if ( f == frame )
    //         return true;
    //     return false;
    // };
    // auto element = std::remove_if( m_keyFrames.begin(), m_keyFrames.end(), find );
    // m_keyFrames.erase( element, m_keyFrames.end() );
    int32_t delIdx = 0;
    for ( auto& fr : m_keyFrames )
    {
        if ( fr == frame )
        {
            for ( auto& feature : fr->m_frameFeatures )
            {
                if ( feature->m_point != nullptr )
                {
                    removeFeature( feature );
                }
            }
            break;
        }
        delIdx++;
    }

    m_keyFrames.erase( m_keyFrames.begin() + delIdx );

    // TODO: also remove from point candidate
}

void Map::removePoint( std::shared_ptr< Point >& point )
{
    // Delete references to mappoints in all keyframes
    for ( auto& feature : point->m_features )
    {
        feature->m_point = nullptr;
        feature->m_frame->removeFeature( feature );
    }
    point->m_features.clear();

    point->m_type = Point::PointType::DELETED;
    // m_trashPoints.push_back( point );
}

void Map::removeFeature( std::shared_ptr< Feature >& feature )
{
    feature->m_point = nullptr;
    if ( feature->m_point->numberObservation() <= 2 )
    {
        removePoint( feature->m_point );
        return;
    }
    // FIXME:  implement removeFeature in point class
    feature->m_point->removeFrame( feature->m_frame );
    feature->m_frame->removeFeature( feature );
}

void Map::addKeyframe( std::shared_ptr< Frame >& frame )
{
    m_keyFrames.push_back( frame );
}

std::shared_ptr< Frame >& Map::getClosestKeyframe( std::shared_ptr< Frame >& frame ) const
{
    std::shared_ptr< Frame > selectedKeyFrame{ nullptr };
    std::vector< keyframeDistance > closeKeyframes;
    getCloseKeyframes( frame, closeKeyframes );
    if ( closeKeyframes.empty() )
    {
        return selectedKeyFrame;
    }

    // TODO: do std::sort instead of this one
    double minDistance = std::numeric_limits< double >::max();
    for ( auto& element : closeKeyframes )
    {
        if ( element.first != frame )
        {
            if ( element.second < minDistance )
            {
                minDistance      = element.second;
                selectedKeyFrame = element.first;
            }
        }
    }

    return selectedKeyFrame;
}

void Map::getCloseKeyframes( const std::shared_ptr< Frame >& frame, std::vector< keyframeDistance >& closeKeyframes ) const
{
    for ( auto& keyFrame : m_keyFrames )
    {
        if ( keyFrame == frame )
            continue;
        // FIXME: m_frameFeatures -> features
        for ( auto& feature : keyFrame->m_frameFeatures )
        {
            if ( feature == nullptr )
            {
                continue;
            }

            if ( frame->isVisible( feature->m_point->m_position ) )
            {
                // TODO: compute the relative pose and get the translation out of that
                closeKeyframes.push_back(
                  std::make_pair( keyFrame, ( frame->m_TransW2F.translation() - keyFrame->m_TransW2F.translation() ).norm() ) );
                break;
            }
        }
    }

    //TODO: sort it here
}

std::shared_ptr< Frame >& Map::getFurthestKeyframe( const Eigen::Vector3d& pos )
{
    std::shared_ptr< Frame > selectedKeyFrame{ nullptr };
    double maxDistance = 0.0;
    for ( auto& frame : m_keyFrames )
    {
        double dist = ( frame->cameraInWorld() - pos ).norm();
        if ( dist > maxDistance )
        {
            maxDistance      = dist;
            selectedKeyFrame = frame;
        }
    }

    return selectedKeyFrame;
}

bool Map::getFrameById( const uint64_t id, std::shared_ptr< Frame >& lookingFrame ) const
{
    for ( const auto& frame : m_keyFrames )
    {
        if ( frame->m_id == id )
        {
            lookingFrame = frame;
            return true;
        }
    }

    return false;
}

void Map::transform( const Eigen::Matrix3d& R, const Eigen::Vector3d& t, const double& s ) const
{
    // TODO: check the formula
    for ( auto& frame : m_keyFrames )
    {
        Eigen::Vector3d pos = s * R * ( frame->cameraInWorld() ) + t;
        Eigen::Matrix3d rot = R * frame->m_TransW2F.rotationMatrix().inverse();
        frame->m_TransW2F   = Sophus::SE3d( rot, pos ).inverse();
        for ( auto& feature : frame->m_frameFeatures )
        {
            // TODO: check the last published ts
            if ( feature->m_point == nullptr )
                continue;
            feature->m_point->m_position = s * R * feature->m_point->m_position + t;
        }
    }
}

std::size_t Map::sizeKeyframes() const
{
    return m_keyFrames.size();
}

void Map::initializeGrid( const std::shared_ptr< PinholeCamera >& camera, const uint32_t cellSize )
{
    m_grid.m_cellSize       = cellSize;
    m_grid.m_gridCols       = ceil( static_cast< double >( camera->width() ) / m_grid.m_cellSize );
    m_grid.m_gridRows       = ceil( static_cast< double >( camera->height() ) / m_grid.m_cellSize );
    const uint32_t numCells = m_grid.m_gridCols * m_grid.m_gridRows;
    m_grid.m_cells.reserve( numCells );

    for ( uint32_t c( 0 ); c < numCells; c++ )
    {
        m_grid.m_cells.emplace_back( std::make_shared< Cell >() );
    }

    m_grid.m_cellOrders.resize( numCells );
    for ( uint32_t i( 0 ); i < numCells; i++ )
    {
        m_grid.m_cellOrders[ i ] = i;
    }

    std::random_device rd;
    std::mt19937 g( rd() );

    // TODO: enable this one again
    std::shuffle( m_grid.m_cellOrders.begin(), m_grid.m_cellOrders.end(), g );  // maybe we should do it at every iteration!
}

void Map::resetGrid()
{
    m_matches = 0;
    m_trials  = 0;
    for ( auto& cell : m_grid.m_cells )
    {
        cell->clear();
    }
}

void Map::reprojectMap( std::shared_ptr< Frame >& frame, std::vector< frameSize >& overlapKeyFrames )
{
    resetGrid();

    // Identify those Keyframes which share a common field of view.
    std::vector< keyframeDistance > closeKeyframes;
    getCloseKeyframes( frame, closeKeyframes );

    // auto compare = [] (const std::pair< const std::shared_ptr< Frame >&, double >& lhs, const std::pair< const std::shared_ptr< Frame >&, double >& rhs) -> bool {
        // return lhs.second < rhs.second;
    // };

    //TODO: Sort KFs with overlap according to their closeness
    // std::sort(closeKeyframes.begin(), closeKeyframes.end(), compare);
    overlapKeyFrames.reserve( 4 );

    for ( const auto& kfDistance : closeKeyframes )
    {
        overlapKeyFrames.push_back( std::make_pair( kfDistance.first, 0 ) );
        for ( const auto& feature : kfDistance.first->m_frameFeatures )
        {
            if ( feature->m_point == nullptr )
                continue;
            if ( feature->m_point->m_lastProjectedKFId == frame->m_id )
                continue;
            feature->m_point->m_lastProjectedKFId = frame->m_id;
            if ( reprojectPoint( frame, feature->m_point ) )
            {
                overlapKeyFrames.back().second++;
            }
        }
    }

    // TODO: Run for candidate

    // Now we go through each grid cell and select one point to match.
    // At the end, we should have at maximum one reprojected point per cell.
    for ( uint32_t i = 0; i < m_grid.m_cells.size(); i++ )
    {
        // we prefer good quality points over unkown quality (more likely to match)
        // and unknown quality over candidates (position not optimized)
        if ( reprojectCell( m_grid.m_cells[ m_grid.m_cellOrders[ i ] ], frame ) )
        {
            m_matches++;
        }
        if ( m_matches > 4 )
        {
            break;
        }
    }
}

bool Map::reprojectCell( std::shared_ptr<Cell>& cell, std::shared_ptr< Frame >& frame )
{
    std::sort( cell->begin(), cell->end(), []( const auto& lhs, const auto& rhs ) -> bool {
        if ( lhs.m_point->m_type > rhs.m_point->m_type )
        {
            return true;
        }
        return false;
    } );

    auto it = cell->begin();
    while ( it != cell->end() )
    {
        m_trials++;

        if ( it->m_point->m_type == Point::PointType::DELETED )
        {
            it = cell->erase( it );
            continue;
        }

        bool foundMatch = algorithm::matchDirect( it->m_point, frame, it->m_feature );
        if ( foundMatch == false )
        {
            // it->pt->n_failed_reproj_++;
            it->m_point->m_failedProjection++;
            if ( it->m_point->m_type == Point::PointType::UNKNOWN && it->m_point->m_failedProjection > 15 )
            {
                removePoint( it->m_point );
            }
            if ( it->m_point->m_type == Point::PointType::CANDIDATE && it->m_point->m_failedProjection > 30 )
            {
                // TODO: remove from candidate
            }
            it = cell->erase( it );
            continue;
        }

        it->m_point->m_succeededProjection++;
        if ( it->m_point->m_type == Point::PointType::UNKNOWN && it->m_point->m_succeededProjection > 10 )
        {
            it->m_point->m_type = Point::PointType::GOOD;
        }

        std::shared_ptr< Feature > feature = std::make_shared< Feature >( frame, it->m_feature, 0 );
        frame->addFeature( feature );
        // Here we add a reference in the feature to the 3D point, the other way
        // round is only done if this frame is selected as keyframe.
        feature->m_point = it->m_point;

        // If the keyframe is selected and we reproject the rest, we don't have to
        // check this point anymore.
        it = cell->erase( it );

        // Maximum one point per cell.
        return true;
    }
    return false;
}

// TODO: check with const point
bool Map::reprojectPoint( const std::shared_ptr< Frame >& frame, const std::shared_ptr< Point >& point )
{
    Eigen::Vector2d pixel = frame->world2image( point->m_position );
    if ( frame->m_camera->isInFrame( pixel, 8 ) )  // 8px is the patch size in the matcher
    {
        const int k =
          static_cast< int >( pixel.y() / m_grid.m_cellSize ) * m_grid.m_gridCols + static_cast< int >( pixel.x() / m_grid.m_cellSize );
        m_grid.m_cells.at( k )->emplace_back( Candidate( point, pixel ) );
        return true;
    }
    return false;
}
