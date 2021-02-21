#include "map.hpp"
#include "algorithm.hpp"

#include <algorithm>
#include <memory>
#include <numeric>
#include <random>
#include <utility>

#include "easylogging++.h"
#define Map_Log( LEVEL ) CLOG( LEVEL, "Map" )

#include <opencv2/video/tracking.hpp>

Map::Map( const std::shared_ptr< PinholeCamera >& camera, const uint32_t cellSize ) : m_matches( 0 ), m_trials( 0 )
{
    initializeGrid( camera, cellSize );
    m_alignment = std::make_shared< FeatureAlignment >( 11, 0, 3 );
}

void Map::reset()
{
    m_keyFrames.clear();
}

void Map::removeFrame( std::shared_ptr< Frame >& frame )
{
    // auto find = [ &frame ]( std::shared_ptr< Frame >& f ) -> bool {
    //     // TODO: check with get() and raw pointer
    //     if ( f.get() == frame.get() )
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
            for ( auto& feature : fr->m_features )
            {
                if ( feature->m_point != nullptr )
                {
                    feature->m_point->removeFrame( feature->m_frame );
                    feature = nullptr;
                }
            }
            break;
        }
        delIdx++;
    }
    frame->m_features.clear();
    frame->m_lastKeyframe = nullptr;
    m_keyFrames.erase( m_keyFrames.begin() + delIdx );

    // TODO: also remove from point candidate
    removeFrameCandidate( frame );
}

void Map::removePoint( std::shared_ptr< Point >& point )
{
    // Delete references to mappoints in all keyframes
    for ( auto& feature : point->m_features )
    {
        feature->m_frame->removeFeature( feature );
    }
    point->m_features.clear();

    point->m_type = Point::PointType::DELETED;
    point         = nullptr;
}

void Map::removeFeature( std::shared_ptr< Feature >& feature )
{
    if (feature->m_point != nullptr)
    {
        if ( feature->m_point->numberObservation() <= 2 )
        {
            removePoint( feature->m_point );
            return;
        }
        feature->m_point->removeFrame( feature->m_frame );
    }
    // FIXME:  implement removeFeature in point class
    feature->m_frame->removeFeature( feature );
    // feature->m_point = nullptr;
    feature = nullptr;
}

void Map::addKeyframe( std::shared_ptr< Frame >& frame )
{
    m_keyFrames.push_back( frame );
}

void Map::getClosestKeyframe( std::shared_ptr< Frame >& frame, std::shared_ptr< Frame >& closestKeyframe ) const
{
    // std::shared_ptr< Frame > selectedKeyFrame{ nullptr };
    std::vector< KeyframeDistance > closeKeyframes;
    getCloseKeyframes( frame, closeKeyframes );
    if ( closeKeyframes.empty() )
    {
        closestKeyframe = nullptr;
    }
    else
    {
        // TODO: do std::sort instead of this one
        double minDistance = std::numeric_limits< double >::max();
        for ( auto& element : closeKeyframes )
        {
            if ( element.first != frame )
            {
                if ( element.second < minDistance )
                {
                    minDistance     = element.second;
                    closestKeyframe = element.first;
                }
            }
        }
    }
}

void Map::getCloseKeyframes( const std::shared_ptr< Frame >& frame, std::vector< KeyframeDistance >& closeKeyframes ) const
{
    // for ( auto& keyFrame : m_keyFrames )
    for ( auto it = m_keyFrames.rbegin(); it != m_keyFrames.rend(); ++it )
    {
        const auto keyFrame = *it;
        if ( keyFrame == frame )
            continue;
        for ( auto& feature : keyFrame->m_features )
        {
            if ( feature == nullptr )
            {
                continue;
            }

            if ( frame->isVisible( feature->m_point->m_position ) )
            {
                // TODO: compute the relative pose and get the translation out of that
                closeKeyframes.push_back(
                  std::make_pair( keyFrame, ( frame->m_absPose.translation() - keyFrame->m_absPose.translation() ).norm() ) );
                break;
            }
        }
    }

    // TODO: sort it here
}

void Map::getFurthestKeyframe( const Eigen::Vector3d& pos, std::shared_ptr< Frame >& furthestKeyframe ) const
{
    double maxDistance = 0.0;
    for ( auto& frame : m_keyFrames )
    {
        double dist = ( frame->cameraInWorld() - pos ).norm();
        if ( dist > maxDistance )
        {
            maxDistance      = dist;
            furthestKeyframe = frame;
        }
    }
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
        Eigen::Matrix3d rot = R * frame->m_absPose.rotationMatrix().inverse();
        frame->m_absPose    = Sophus::SE3d( rot, pos ).inverse();
        for ( auto& feature : frame->m_features )
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
        m_grid.m_cells.emplace_back( std::make_shared< CandidateList >() );
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

void Map::reprojectMap( const std::shared_ptr< Frame >& refFrame,
                        std::shared_ptr< Frame >& curFrame,
                        std::vector< FrameSize >& overlapKeyFrames )
{
    resetGrid();

    std::vector< bool > visGrid( m_grid.m_gridCols * m_grid.m_gridRows, false );

    std::vector< cv::Point2f > refPoints;
    refPoints.reserve( 200 );
    std::vector< cv::Point2f > curPoints;
    curPoints.reserve( 200 );
    std::vector< std::shared_ptr< Point > > selectedPoints;

    // const Sophus::SE3d relativePose1 = algorithm::computeRelativePose( refFrame, curFrame );
    for ( auto& feature : refFrame->m_features )
    {
        if ( feature->m_point != nullptr )
        {
            auto& point               = feature->m_point;
            const auto& pixelPosition = feature->m_pixelPosition;
            // const int32_t k           = static_cast< int32_t >( pixelPosition.y() ) / m_grid.m_cellSize * m_grid.m_gridCols +
            //                   static_cast< int32_t >( pixelPosition.x() ) / m_grid.m_cellSize;
            // if ( visGrid[ k ] == false )
            // {
            Eigen::Vector2d candidatePixelPosition = curFrame->world2image( point->m_position );
            // double error                           = m_alignment->align( feature, curFrame, candidatePixelPosition );
            // if ( curFrame->m_camera->isInFrame( candidatePixelPosition, 4 ) && error < 30 )
            // {
            //     std::shared_ptr< Feature > newFeature = std::make_shared< Feature >( curFrame, candidatePixelPosition, 0 );
            //     curFrame->addFeature( newFeature );
            //     // Here we add a reference in the feature to the 3D point, the other way
            //     // round is only done if this frame is selected as keyframe.
            //     newFeature->setPoint( point );
            //     point->addFeature( newFeature );
            //     Map_Log( DEBUG ) << "feature accepted with error: " << error;
            //     visGrid[ k ] = true;
            // }

            refPoints.emplace_back( cv::Point2f( static_cast< float >( feature->m_pixelPosition.x() ),
                                                    static_cast< float >( feature->m_pixelPosition.y() ) ) );
            curPoints.emplace_back(
                cv::Point2f( static_cast< float >( candidatePixelPosition.x() ), static_cast< float >( candidatePixelPosition.y() ) ) );
            selectedPoints.push_back( point );
            // }
        }
    }

    {
        const cv::Mat& refImg         = refFrame->m_imagePyramid.getBaseImage();
        const cv::Mat& curImg         = curFrame->m_imagePyramid.getBaseImage();
        const uint64_t refObservation = refPoints.size();

        std::vector< uchar > status;
        status.reserve( refObservation );
        std::vector< float > errors;
        errors.reserve( refObservation );
        const int maxIteration    = 30;
        const double epsilonError = 1e-4;

        cv::TermCriteria termcrit( cv::TermCriteria::COUNT + cv::TermCriteria::EPS, maxIteration, epsilonError );
        cv::calcOpticalFlowPyrLK( refImg, curImg, refPoints, curPoints, status, errors, cv::Size( 11, 11 ), 3, termcrit,
                                  cv::OPTFLOW_USE_INITIAL_FLOW );

        uint32_t cntAdded = 0;
        for ( std::size_t i( 0 ); i < curPoints.size(); i++ )
        {
            if ( status[ i ] == true )
            {
                const Eigen::Vector2d pixelPosition( curPoints[ i ].x, curPoints[ i ].y );
                const int32_t k = static_cast< int32_t >( pixelPosition.y() ) / m_grid.m_cellSize * m_grid.m_gridCols +
                                    static_cast< int32_t >( pixelPosition.x() ) / m_grid.m_cellSize;
                if ( curFrame->m_camera->isInFrame( pixelPosition, 4 ) && visGrid[ k ] == false )
                {
                    std::shared_ptr< Feature > newFeature =
                      std::make_shared< Feature >( curFrame, Eigen::Vector2d( pixelPosition.x(), pixelPosition.y() ), 0 );
                    curFrame->addFeature( newFeature );
                    // Here we add a reference in the feature to the 3D point, the other way
                    // round is only done if this frame is selected as keyframe.
                    newFeature->setPoint( selectedPoints[ i ] );
                    selectedPoints[ i ]->addFeature( newFeature );
                    visGrid[ k ] = true;
                    cntAdded++;
                }
            }
        }
        Map_Log( DEBUG ) << "number of added with last frame " << cntAdded;
    }

    refPoints.clear();
    refPoints.reserve( 200 );
    curPoints.clear();
    curPoints.reserve( 200 );
    selectedPoints.clear();

    for ( auto& feature : refFrame->m_lastKeyframe->m_features )
    {
        if ( feature->m_point != nullptr )
        {
            auto& point               = feature->m_point;
            if (point->findFrame (curFrame) == false)
            {
                const auto& pixelPosition = feature->m_pixelPosition;
                // const int32_t k           = static_cast< int32_t >( pixelPosition.y() ) / m_grid.m_cellSize * m_grid.m_gridCols +
                //                   static_cast< int32_t >( pixelPosition.x() ) / m_grid.m_cellSize;
                // if ( visGrid[ k ] == false )
                // {
                Eigen::Vector2d candidatePixelPosition = curFrame->world2image( point->m_position );
                // double error                           = m_alignment->align( feature, curFrame, candidatePixelPosition );
                // if ( curFrame->m_camera->isInFrame( candidatePixelPosition, 4 ) && error < 50 )
                // {
                //     std::shared_ptr< Feature > newFeature = std::make_shared< Feature >( curFrame, candidatePixelPosition, 0 );
                //     curFrame->addFeature( newFeature );
                //     // Here we add a reference in the feature to the 3D point, the other way
                //     // round is only done if this frame is selected as keyframe.
                //     newFeature->setPoint( point );
                //     point->addFeature( newFeature );
                //     Map_Log( DEBUG ) << "feature accepted with error: " << error;
                //     visGrid[ k ] = true;
                // }
                refPoints.emplace_back( cv::Point2f( static_cast< float >( feature->m_pixelPosition.x() ),
                                                        static_cast< float >( feature->m_pixelPosition.y() ) ) );
                curPoints.emplace_back(
                    cv::Point2f( static_cast< float >( candidatePixelPosition.x() ), static_cast< float >( candidatePixelPosition.y() ) ) );
                selectedPoints.push_back( point );
            }
        }
    }

    {
        const cv::Mat& refImg         = refFrame->m_lastKeyframe->m_imagePyramid.getBaseImage();
        const cv::Mat& curImg         = curFrame->m_imagePyramid.getBaseImage();
        const uint64_t refObservation = refPoints.size();

        std::vector< uchar > status;
        status.reserve( refObservation );
        std::vector< float > errors;
        errors.reserve( refObservation );
        const int maxIteration    = 30;
        const double epsilonError = 1e-4;

        cv::TermCriteria termcrit( cv::TermCriteria::COUNT + cv::TermCriteria::EPS, maxIteration, epsilonError );
        cv::calcOpticalFlowPyrLK( refImg, curImg, refPoints, curPoints, status, errors, cv::Size( 11, 11 ), 3, termcrit,
                                  cv::OPTFLOW_USE_INITIAL_FLOW );

        uint32_t cntAdded = 0;
        for ( std::size_t i( 0 ); i < curPoints.size(); i++ )
        {
            if ( status[ i ] == true )
            {
                const Eigen::Vector2d pixelPosition( curPoints[ i ].x, curPoints[ i ].y );
                const int32_t k = static_cast< int32_t >( pixelPosition.y() ) / m_grid.m_cellSize * m_grid.m_gridCols +
                                    static_cast< int32_t >( pixelPosition.x() ) / m_grid.m_cellSize;
                if ( curFrame->m_camera->isInFrame( pixelPosition, 4 ) && visGrid[ k ] == false )
                {
                    std::shared_ptr< Feature > newFeature =
                      std::make_shared< Feature >( curFrame, Eigen::Vector2d( pixelPosition.x(), pixelPosition.y() ), 0 );
                    curFrame->addFeature( newFeature );
                    // Here we add a reference in the feature to the 3D point, the other way
                    // round is only done if this frame is selected as keyframe.
                    newFeature->setPoint( selectedPoints[ i ] );
                    selectedPoints[ i ]->addFeature( newFeature );
                    visGrid[ k ] = true;
                    cntAdded++;
                }
            }
        }
        Map_Log( DEBUG ) << "number of added with last last frame " << cntAdded;
    }

    /*

    // Identify those Keyframes which share a common field of view.
    std::vector< KeyframeDistance > closeKeyframes;
    // getCloseKeyframes( curFrame, closeKeyframes );
    closeKeyframes.push_back(
      std::make_pair( refFrame, ( curFrame->m_absPose.translation() - refFrame->m_absPose.translation() ).norm() ) );
    closeKeyframes.push_back( std::make_pair(
      refFrame->m_lastKeyframe, ( curFrame->m_absPose.translation() - refFrame->m_lastKeyframe->m_absPose.translation() ).norm() ) );

    // auto compare = [] (const std::pair< const std::shared_ptr< Frame >&, double >& lhs, const std::pair< const std::shared_ptr< Frame >&,
    // double >& rhs) -> bool {
    //     return lhs.second < rhs.second;
    // };

    // TODO: Sort KFs with overlap according to their closeness
    // std::sort(closeKeyframes.begin(), closeKeyframes.end(), compare);
    overlapKeyFrames.reserve( 4 );
    int32_t n = 0;

    // for ( const auto& kfDistance : closeKeyframes )
    for ( uint32_t i( 0 ); i < closeKeyframes.size() && n < 4; i++, n++ )
    {
        const auto& kfDistance = closeKeyframes[ i ];
        overlapKeyFrames.push_back( std::make_pair( kfDistance.first, 0 ) );
        Map_Log( DEBUG ) << "Frame id " << kfDistance.first->m_id << " projected to find the features";
        // project 3d points from closeKeyFrame into new frame. If we found some projected points, we update the frame ID of 3d point.
        for ( const auto& feature : kfDistance.first->m_features )
        {
            if ( feature->m_point == nullptr )
                continue;
            if ( feature->m_point->m_lastProjectedKFId == curFrame->m_id )
                continue;
            feature->m_point->m_lastProjectedKFId = curFrame->m_id;
            if ( reprojectPoint( curFrame, feature ) )
            {
                overlapKeyFrames.back().second++;
            }
        }
    }

    // Now we go through each grid cell and select one point to match.
    // At the end, we should have at maximum one reprojected point per cell.
    for ( uint32_t i = 0; i < m_grid.m_cells.size(); i++ )
    {
        // we prefer good quality points over unkown quality (more likely to match)
        // and unknown quality over candidates (position not optimized)
        const uint32_t idx                           = m_grid.m_cellOrders[ i ];
        std::shared_ptr< CandidateList >& candidates = m_grid.m_cells[ idx ];
        if ( candidates->size() > 0 && reprojectCell( candidates, curFrame ) )
        {
            m_matches++;
        }
        if ( m_matches > 120 )
        {
            break;
        }
    }
    */
}

// TODO: check with const point
bool Map::reprojectPoint( const std::shared_ptr< Frame >& frame, const std::shared_ptr< Feature >& feature )
{
    const Eigen::Vector2d pixel = frame->world2image( feature->m_point->m_position );
    if ( frame->m_camera->isInFrame( pixel, 3 ) )  // 8px is the patch size in the matcher
    {
        const int32_t k = static_cast< int32_t >( pixel.y() ) / m_grid.m_cellSize * m_grid.m_gridCols +
                          static_cast< int32_t >( pixel.x() ) / m_grid.m_cellSize;
        m_grid.m_cells[ k ]->emplace_back( Candidate( feature, feature->m_point, frame ) );
        return true;
    }
    return false;
}

bool Map::reprojectCell( std::shared_ptr< CandidateList >& candidates, std::shared_ptr< Frame >& frame )
{
    std::sort( candidates->begin(), candidates->end(), []( const auto& lhs, const auto& rhs ) -> bool {
        if ( lhs.m_point->m_type > rhs.m_point->m_type )
        {
            return true;
        }
        return false;
    } );

    for ( auto& candidate : *candidates )
    {
        m_trials++;
        auto& point = candidate.m_point;

        if ( point->m_type == Point::PointType::DELETED )
        {
            // it = cell->erase( it );
            continue;
        }

        // bool foundMatch = algorithm::matchDirect( candidate.m_point, frame, candidate.m_feature );

        // TODO: add corresponding feature to cell. point, feature in previous frame and pixel position
        // std::shared_ptr< Feature > refFeature;
        // if ( candidate.m_point->getCloseViewObservation( frame->cameraInWorld(), refFeature ) == false )
        // {
        //     continue;
        // }

        Eigen::Vector2d candidatePixelPosition = frame->world2image( point->m_position );
        // Map_Log( DEBUG ) << "pixel pos: " << candidatePixelPosition.transpose()
        //                  << ", init error: " << algorithm::computePatchError( candidate.m_feature, frame, candidatePixelPosition, 7 );
        // double error = m_alignment->align( candidate.m_feature, frame, candidatePixelPosition );
        // bool foundMatch = error < 50.0 ? true : false;
        // Map_Log( DEBUG ) << "error: " << error << ", update pixel pos: " << candidatePixelPosition.transpose()
        //                  << ", final error: " << algorithm::computePatchError( candidate.m_feature, frame, candidatePixelPosition, 7 );

        // // TODO: check the projected area with reference and compute the error

        // if ( foundMatch == false )
        // {
        //     point->m_failedProjection++;
        //     if ( point->m_type == Point::PointType::UNKNOWN && point->m_failedProjection > 15 )
        //     {
        //         removePoint( point );
        //     }
        //     if ( point->m_type == Point::PointType::CANDIDATE && point->m_failedProjection > 30 )
        //     {
        //         // TODO: remove from candidate
        //     }
        //     // it = cell->erase( it );
        //     continue;
        // }

        point->m_succeededProjection++;
        if ( point->m_type == Point::PointType::UNKNOWN && point->m_succeededProjection > 10 )
        {
            point->m_type = Point::PointType::GOOD;
        }

        std::shared_ptr< Feature > feature = std::make_shared< Feature >( frame, candidatePixelPosition, 0 );
        frame->addFeature( feature );
        // Here we add a reference in the feature to the 3D point, the other way
        // round is only done if this frame is selected as keyframe.
        feature->setPoint( point );
        point->addFeature( feature );
        Map_Log( DEBUG ) << "feature accepted";

        // Maximum one point per cell.
        return true;
    }
    return false;
}

// const std::vector< Map::Candidate >& Map::getCandidateList () const
// {
//     return m_candidates;
// }

void Map::addNewCandidate( const std::shared_ptr< Feature >& feature,
                           const std::shared_ptr< Point >& point,
                           const std::shared_ptr< Frame >& visitedFrame )
{
    point->m_type = Point::PointType::CANDIDATE;
    std::unique_lock< std::mutex > lock( m_mutexCandidates );
    m_candidates.push_back( Candidate( feature, point, visitedFrame ) );
}

void Map::addCandidateToFrame( std::shared_ptr< Frame >& frame )
{
    std::unique_lock< std::mutex > lock( m_mutexCandidates );
    uint32_t cntAddedFeature = 0;
    for ( auto& candidate : m_candidates )
    {
        if ( candidate.m_feature->m_frame == frame )
        {
            candidate.m_feature->setPoint( candidate.m_point );
            candidate.m_point->addFeature( candidate.m_feature );
            candidate.m_point->m_type             = Point::PointType::UNKNOWN;
            candidate.m_point->m_failedProjection = 0;
            cntAddedFeature++;
        }
    }
    Map_Log( DEBUG ) << cntAddedFeature << " features added to frame id " << frame->m_id;
    removeFrameCandidate( frame );
}

void Map::addCandidateToAllActiveKeyframes()
{
    std::unique_lock< std::mutex > lock( m_mutexCandidates );
    for ( auto& keyframe : m_keyFrames )
    {
        uint32_t cntAddedFeature = 0;
        for ( auto& candidate : m_candidates )
        {
            if ( candidate.m_feature->m_frame == keyframe )
            {
                candidate.m_feature->setPoint( candidate.m_point );
                candidate.m_point->addFeature( candidate.m_feature );

                // if (candidate.m_visitedFrame->isKeyframe() == false)
                // {
                //     const Eigen::Vector2d pixelPosition = candidate.m_visitedFrame->world2image( candidate.m_point->m_position );
                //     std::shared_ptr< Feature > feature =
                //     std::make_shared< Feature >( candidate.m_visitedFrame, pixelPosition, 0.0, 0.0, 0, Feature::FeatureType::EDGE );
                //     candidate.m_visitedFrame->addFeature( feature );
                //     feature->setPoint( candidate.m_point );
                //     candidate.m_point->addFeature( feature );
                // }

                candidate.m_point->m_type             = Point::PointType::UNKNOWN;
                candidate.m_point->m_failedProjection = 0;
                cntAddedFeature++;
            }
        }
        Map_Log( DEBUG ) << cntAddedFeature << " features added to frame id " << keyframe->m_id;
        removeFrameCandidate( keyframe );
    }
}

void Map::removeFrameCandidate( std::shared_ptr< Frame >& frame )
{
    // std::unique_lock< std::mutex > lock( m_mutexCandidates );
    m_candidates.erase( std::remove_if( m_candidates.begin(), m_candidates.end(),
                                        [ &frame ]( const auto& candidate ) { return candidate.m_feature->m_frame.get() == frame.get(); } ),
                        m_candidates.end() );
}

void Map::resetCandidates()
{
    m_candidates.clear();
}