/**
 * @file depth_Estimator.hpp
 * @brief gaussian distribution representation for depth
 *
 * @date 22.04.2020
 * @author Amin Abouee
 *
 * @section DESCRIPTION
 *
 *
 */
#ifndef __DEPTH_ESTIMATOR_H__
#define __DEPTH_ESTIMATOR_H__

#include "feature_selection.hpp"
#include "frame.hpp"
#include "mixed_gaussian_filter.hpp"
#include "map.hpp"

#include <Eigen/Core>
#include <sophus/se3.hpp>

#include <condition_variable>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>


class DepthEstimator final
{
public:
    std::unique_ptr< std::thread > m_thread;
    std::mutex m_mutexFilter;
    std::mutex m_mutexFrame;
    std::condition_variable m_condition;
    // std::unique_lock<std::mutex> m_threadLocker;

    std::shared_ptr< Frame > m_newKeyframe;
    std::shared_ptr< Frame > m_deletedKeyframe;
    std::queue< std::shared_ptr< Frame > > m_queueFrames;
    std::vector< MixedGaussianFilter > m_depthFilters;

    std::shared_ptr< Map > m_map;
    std::shared_ptr< FeatureSelection > m_featureSelector;

    double m_newKeyframeMinDepth;
    double m_newKeyframeMeanDepth;
    bool m_haltUpdatingDepthFilter;
    // bool m_haveDeletedKeyFrame;
    bool m_newKeyframeAdded;
    bool m_activeThread;
    bool m_terminateThread;

    // C'tor
    explicit DepthEstimator(std::shared_ptr<Map>& map,  std::shared_ptr< FeatureSelection >& featureSelection);
    // Copy C'tor
    DepthEstimator( const DepthEstimator& rhs ) = delete;
    // move C'tor
    DepthEstimator( DepthEstimator&& rhs ) = delete;
    // Copy assignment operator
    DepthEstimator& operator=( const DepthEstimator& rhs ) = delete;
    // move assignment operator
    DepthEstimator& operator=( DepthEstimator&& rhs ) = delete;
    // D'tor
    ~DepthEstimator();

    /// Add frame to the queue to be processed.
    void addFrame( std::shared_ptr< Frame >& frame );

    /// Add new keyframe to the queue
    void addKeyframe( std::shared_ptr< Frame >& frame, double depthMean, double depthMin );

    /// Remove all seeds which are initialized from the specified keyframe. This
    /// function is used to make sure that no seeds points to a non-existent frame
    /// when a frame is removed from the map.
    void removeKeyframe( std::shared_ptr< Frame >& frame );

    uint32_t numberFilters () const;

    /// If the map is reset, call this function such that we don't have pointers
    /// to old frames.
    void reset();

private:

    /// Bayes update of the seed, x is the measurement, tau2 the measurement uncertainty
    /// Reference: Video-based, real-time multi-view stereo. Supplementary matterial
    void updateFilter( const double x, const double tau2, MixedGaussianFilter& depthFilter );

    void removeKeyframe();

    /// Compute the uncertainty of the measurement.
    double computeTau( const Sophus::SE3d& relativePose, const Eigen::Vector3d& bearing, const double z, const double pixelErrorAngle );

    /// Initialize new Filters from a frame.
    void initializeFilters( std::shared_ptr< Frame >& frame );

    /// Update all Filters with a new measurement frame.
    void updateFilters( std::shared_ptr< Frame >& frame );

    /// When a new keyframe arrives, the frame queue should be cleared.
    void clearFrameQueue();

    /// A thread that is continuously updating the seeds.
    void updateFiltersLoop();

private:
};

#endif /* __DEPTH_ESTIMATOR_H__ */