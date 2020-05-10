/**
 * @file depth_filter.hpp
 * @brief gaussian distribution representation for depth
 *
 * @date 22.04.2020
 * @author Amin Abouee
 *
 * @section DESCRIPTION
 *
 *
 */
#ifndef __DEPTH_FILTER_H__
#define __DEPTH_FILTER_H__

#include <condition_variable>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>

#include "frame.hpp"

#include <Eigen/Core>
#include <sophus/se3.hpp>

class DepthFilter final
{
public:
    std::unique_ptr< std::thread > m_thread;
    std::mutex m_mutexSeed;
    std::mutex m_mutexFrame;
    std::condition_variable m_condition;
    // std::unique_lock<std::mutex> m_threadLocker;

    std::shared_ptr< Frame > m_newKeyframe;
    std::queue< std::shared_ptr< Frame > > m_queueFrames;

    bool m_haltUpdatingSeed;
    bool m_newKeyframeAdded;
    bool m_activeThread;
    double m_newKeyframeMinDepth;
    double m_newKeyframeMeanDepth;

    // C'tor
    explicit DepthFilter();
    // Copy C'tor
    DepthFilter( const DepthFilter& rhs ) = delete;
    // move C'tor
    DepthFilter( DepthFilter&& rhs ) = delete;
    // Copy assignment operator
    DepthFilter& operator=( const DepthFilter& rhs ) = delete;
    // move assignment operator
    DepthFilter& operator=( DepthFilter&& rhs ) = delete;
    // D'tor
    ~DepthFilter();

    /// Add frame to the queue to be processed.
    void addFrame( std::shared_ptr< Frame >& frame );

    /// Add new keyframe to the queue
    void addKeyframe( std::shared_ptr< Frame >& frame, double depthMean, double depthMin );

    /// Remove all seeds which are initialized from the specified keyframe. This
    /// function is used to make sure that no seeds points to a non-existent frame
    /// when a frame is removed from the map.
    void removeKeyframe( std::shared_ptr< Frame >& frame );

    /// If the map is reset, call this function such that we don't have pointers
    /// to old frames.
    void reset();

    /// Compute the uncertainty of the measurement.
    double computeTau( const Sophus::SE3d& T_ref_cur, const Eigen::Vector3d& f, const double z, const double px_error_angle );

    /// Initialize new seeds from a frame.
    void initializeSeeds( std::shared_ptr< Frame >& frame );

    /// Update all seeds with a new measurement frame.
    void updateSeeds( std::shared_ptr< Frame >& frame );

    /// When a new keyframe arrives, the frame queue should be cleared.
    void clearFrameQueue();

    /// A thread that is continuously updating the seeds.
    void updateSeedsLoop();

private:
};

#endif /* __DEPTH_FILTER_H__ */