#include "map.hpp"
#include <numeric>

#include "easylogging++.h"
#define Map_Log( LEVEL ) CLOG( LEVEL, "Map" )

Map::Map()
{
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
    if ( feature->m_point->m_features.size() <= 2 )
    {
        removePoint( feature->m_point );
        return;
    }
    feature->m_point = nullptr;
    feature->m_point->removeFrame( feature->m_frame );
    feature->m_frame->removeFeature( feature );
}

void Map::addKeyframe( std::shared_ptr< Frame >& frame )
{
    m_keyFrames.push_back( frame );
}

std::shared_ptr< Frame >& Map::getCloseKeyframe( std::shared_ptr< Frame >& frame ) const
{
    std::shared_ptr< Frame > selectedKeyFrame{ nullptr };
    std::vector< keyframeDistance > closeKeyframes;
    getCloseKeyframes( frame, closeKeyframes );
    if ( closeKeyframes.empty() )
    {
        return selectedKeyFrame;
    }

    // std::sort( closeKeyframes.begin(), closeKeyframes.end(),
    //            []( const auto& left, const auto& right ) { return left.second < right.second; } );

    double minDistance = std::numeric_limits<double>::max();
    for ( auto& element : closeKeyframes )
    {
        if (element.first != frame)
        {
            if (element.second < minDistance)
            {
                minDistance = element.second;
                selectedKeyFrame = element.first;
            }
        }
    }

    return selectedKeyFrame;

    // if ( closeKeyframes[ 0 ].first != frame )
    // {
    //     selectedKeyFrame = closeKeyframes[ 0 ].first;
    // }
    // else
    // {
    //     selectedKeyFrame = closeKeyframes[ 1 ].first;
    // }

    return selectedKeyFrame;
}

void Map::getCloseKeyframes( const std::shared_ptr< Frame >& frame, std::vector< keyframeDistance >& closeKeyframes ) const
{
    for ( auto& keyFrame : m_keyFrames )
    {
        if ( keyFrame == frame )
            continue;
        for ( auto& feature : keyFrame->m_frameFeatures )
        {
            if ( frame->isVisible( feature->m_point->m_position ) )
            {
                closeKeyframes.push_back(
                  std::make_pair( keyFrame, ( frame->m_TransW2F.translation() - keyFrame->m_TransW2F.translation() ).norm() ) );
                break;
            }
        }
    }
}

void Map::transform( const Eigen::Matrix3d& R, const Eigen::Vector3d& t, const double& s ) const
{
    //     for(auto it=keyframes_.begin(), ite=keyframes_.end(); it!=ite; ++it)
    //   {
    //     Vector3d pos = s*R*(*it)->pos() + t;
    //     Matrix3d rot = R*(*it)->T_f_w_.rotation_matrix().inverse();
    //     (*it)->T_f_w_ = SE3(rot, pos).inverse();
    //     for(auto ftr=(*it)->fts_.begin(); ftr!=(*it)->fts_.end(); ++ftr)
    //     {
    //       if((*ftr)->point == NULL)
    //         continue;
    //       if((*ftr)->point->last_published_ts_ == -1000)
    //         continue;
    //       (*ftr)->point->last_published_ts_ = -1000;
    //       (*ftr)->point->pos_ = s*R*(*ftr)->point->pos_ + t;
    //     }
    //   }

    // TODO: check the formula
    for ( auto& frame : m_keyFrames )
    {
        Eigen::Vector3d pos = s * R * ( frame->cameraInWorld() ) + t;
        Eigen::Matrix3d rot = R * frame->m_TransW2F.rotationMatrix().inverse();
        frame->m_TransW2F = Sophus::SE3d(rot, pos).inverse();
        for ( auto& feature : frame->m_frameFeatures )
        {
            // TODO: check the last published ts
            if ( feature->m_point == nullptr )
                continue;
            feature->m_point->m_position = s * R * feature->m_point->m_position + t;
        }
    }
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

bool Map::getFrameById( const uint64_t id, const std::shared_ptr< Frame >& frame ) const
{
    for ( const auto& frame : m_keyFrames )
    {
        if ( frame->m_id == id )
        {
            return true;
        }
    }

    return false;
}

std::size_t Map::sizeKeyframes() const
{
    return m_keyFrames.size();
}
