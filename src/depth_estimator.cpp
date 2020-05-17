#include "depth_estimator.hpp"
#include "algorithm.hpp"
#include "utils.hpp"

#include <random>

#include "easylogging++.h"
#define Map_Log( LEVEL ) CLOG( LEVEL, "DepthEstimator" )

DepthEstimator::DepthEstimator()
    : m_haltUpdatingDepthFilter( false ), m_activeThread( false ), m_newKeyframeMinDepth( 0.0 ), m_newKeyframeMeanDepth( 0.0 )
{
    // https://stackoverflow.com/a/18376082/1804533
    // https://thispointer.com/c-11-multithreading-part-1-three-different-ways-to-create-threads/
    // new std::thread(std::bind(&ThreadExample::run, this)));
    m_thread = std::make_unique< std::thread >( &DepthEstimator::updateFiltersLoop, this );
}

DepthEstimator::~DepthEstimator()
{
    if ( m_thread != nullptr && m_thread->joinable() )
    {
        m_haltUpdatingDepthFilter = true;
        // m_thread_->interrupt();
        m_activeThread = false;
        m_thread->join();
        // thread_ = NULL;
        m_thread = nullptr;
    }
}

void DepthEstimator::addFrame( std::shared_ptr< Frame >& frame )
{
    if ( m_thread != nullptr )
    {
        std::unique_lock< std::mutex > threadLocker( m_mutexFrame );
        if ( m_queueFrames.size() > 2 )
            m_queueFrames.pop();
        m_queueFrames.push( frame );
        m_haltUpdatingDepthFilter = false;
        m_condition.notify_one();
    }
    else
    {
        updateFilters( frame );
    }
}

void DepthEstimator::addKeyframe( std::shared_ptr< Frame >& frame, double depthMean, double depthMin )
{
    m_newKeyframeMinDepth = depthMin;
    m_newKeyframeMinDepth = depthMin;
    if ( m_thread != nullptr )
    {
        m_newKeyframe             = frame;
        m_haltUpdatingDepthFilter = true;
        m_newKeyframeAdded        = true;
        m_condition.notify_one();
    }
    else
    {
        initializeFilters( frame );
    }
}

void DepthEstimator::removeKeyframe( std::shared_ptr< Frame >& frame )
{
    m_haltUpdatingDepthFilter = true;
    std::unique_lock< std::mutex > threadLocker( m_mutexSeed );
    // std::list<Seed>::iterator it=seeds_.begin();
    // size_t n_removed = 0;
    // while(it!=seeds_.end())
    // {
    //     if(it->ftr->frame == frame.get())
    //     {
    //     it = seeds_.erase(it);
    //     ++n_removed;
    //     }
    //     else
    //     ++it;
    // }
    m_haltUpdatingDepthFilter = false;
}

void DepthEstimator::reset()
{
    m_haltUpdatingDepthFilter = true;
    std::unique_lock< std::mutex > threadLocker( m_mutexSeed );
    // seeds_.clear();
    // TODO: check this line
    // std::unique_lock< std::mutex > threadLocker();
    while ( !m_queueFrames.empty() )
        m_queueFrames.pop();
    m_haltUpdatingDepthFilter = false;
}

void DepthEstimator::initializeFilters( std::shared_ptr< Frame >& frame )
{
    // detect new feature

    // lock mutex
    // load the feature

    // create seeds

    // set halt updating seed

    m_haltUpdatingDepthFilter = true;
    std::unique_lock< std::mutex > threadLocker( m_mutexSeed );  // by locking the updateSeeds function stops
    // ++Seed::batch_counter;
    //   std::for_each(new_features.begin(), new_features.end(), [&](Feature* ftr){
    //     seeds_.push_back(Seed(ftr, new_keyframe_mean_depth_, new_keyframe_min_depth_));
    //   });

    for ( const auto& feature : frame->m_frameFeatures )
    {
        m_depthFilters.emplace_back( MixedGaussianFilter( feature, m_newKeyframeMeanDepth, m_newKeyframeMinDepth ) );
    }

    m_haltUpdatingDepthFilter = false;
}

void DepthEstimator::updateFilters( std::shared_ptr< Frame >& frame )
{
    //     // update only a limited number of seeds, because we don't have time to do it
    //     // for all the seeds in every frame!
    //   size_t n_updates=0, n_failed_matches=0, n_seeds = seeds_.size();
    //   lock_t lock(seeds_mut_);
    //   std::list<Seed>::iterator it=seeds_.begin();

    // const double focal_length = frame->cam_->errorMultiplier2();
    const double focal_length   = 10;
    const double px_noise       = 1.0;
    const double px_error_angle = atan( px_noise / ( 2.0 * focal_length ) ) * 2.0;  // law of chord (sehnensatz)

    for ( auto& depthFilter : m_depthFilters )
    {
        if ( m_haltUpdatingDepthFilter == false )
            return;

        // if the current depth filter is very old, delete it
        if ( MixedGaussianFilter::m_frameCounter - depthFilter.m_frameCounter > 2 )
        {
            depthFilter.m_validity = false;
            continue;
        }

        const Sophus::SE3d relativePose        = algorithm::computeRelativePose( depthFilter.m_feature->m_frame, frame );
        const Eigen::Vector3d pointInCurCamera = relativePose * ( depthFilter.m_feature->m_bearingVec / depthFilter.m_mu );
        if ( pointInCurCamera.z() < 0 )
        {
            depthFilter.m_validity = false;
            continue;
        }
        if ( frame->isVisible( pointInCurCamera ) == false )
        {
            depthFilter.m_validity = false;
            continue;
        }

        const double inverseMinDepth = depthFilter.m_mu + depthFilter.m_sigma;
        const double inverseMaxDepth = std::max( depthFilter.m_mu - depthFilter.m_sigma, 1e-7 );
        const double depth           = 0.0;

        // check in epipolar distance

        // compute tau
        const double tau         = computeTau( relativePose, depthFilter.m_feature->m_bearingVec, depth, px_error_angle );
        const double tau_inverse = 0.5 * ( 1.0 / std::max( 1e-7, depth - tau ) - 1.0 / ( depth + tau ) );

        // update the estimate
        updateFilter( 1. / depth, tau_inverse * tau_inverse, depthFilter );

        // if the seed has converged, we initialize a new candidate point and remove the seed
        if ( sqrt( depthFilter.m_var ) < depthFilter.m_maxDepth / 10.0 )
        {
            // assert( it->ftr->point == NULL );  // TODO this should not happen anymore
            const Eigen::Vector3d pointInWorld =
              depthFilter.m_feature->m_frame->image2camera( depthFilter.m_feature->m_feature, 1.0 / depthFilter.m_mu );
            // Point* point   = new Point( xyz_world, it->ftr );
            // it->ftr->point = point;
            /* FIXME it is not threadsafe to add a feature to the frame here.
            if(frame->isKeyframe())
            {
              Feature* ftr = new Feature(frame.get(), matcher_.px_cur_, matcher_.search_level_);
              ftr->point = point;
              point->addFrameRef(ftr);
              frame->addFeature(ftr);
              it->ftr->frame->addFeature(it->ftr);
            }
            else
            */
            {
                // seed_converged_cb_( point, it->sigma2 );  // put in candidate list
            }
            // it = seeds_.erase( it );
            depthFilter.m_validity = false;
        }
        else if ( std::isnan( inverseMinDepth ) )
        {
            depthFilter.m_validity = false;
        }
    }
}

void DepthEstimator::updateFilter( const double x, const double tau2, MixedGaussianFilter& depthFilter )
{
    const double norm_scale = sqrt( depthFilter.m_var + tau2 );
    if ( std::isnan( norm_scale ) )
        return;
    // std::normal_distribution< double > normalDist( depthFilter.m_mu, norm_scale );
    const double s2 = 1. / ( 1. / depthFilter.m_var + 1. / tau2 );
    const double m  = s2 * ( depthFilter.m_mu / depthFilter.m_var + x / tau2 );
    // https://stackoverflow.com/questions/10847007/using-the-gaussian-probability-density-function-in-c
    double C1 =
      depthFilter.m_a / ( depthFilter.m_a + depthFilter.m_b ) * algorithm::computeNormalDistribution( depthFilter.m_mu, norm_scale, x );
    double C2                           = depthFilter.m_b / ( depthFilter.m_a + depthFilter.m_b ) * 1. / depthFilter.m_maxDepth;
    const double normalization_constant = C1 + C2;
    C1 /= normalization_constant;
    C2 /= normalization_constant;
    const double f = C1 * ( depthFilter.m_a + 1. ) / ( depthFilter.m_a + depthFilter.m_b + 1. ) +
                     C2 * depthFilter.m_a / ( depthFilter.m_a + depthFilter.m_b + 1. );
    const double e = C1 * ( depthFilter.m_a + 1. ) * ( depthFilter.m_a + 2. ) /
                       ( ( depthFilter.m_a + depthFilter.m_b + 1. ) * ( depthFilter.m_a + depthFilter.m_b + 2. ) ) +
                     C2 * depthFilter.m_a * ( depthFilter.m_a + 1.0f ) /
                       ( ( depthFilter.m_a + depthFilter.m_b + 1.0f ) * ( depthFilter.m_a + depthFilter.m_b + 2.0f ) );

    // update parameterss
    const double mu_new = C1 * m + C2 * depthFilter.m_mu;
    depthFilter.m_var   = C1 * ( s2 + m * m ) + C2 * ( depthFilter.m_var + depthFilter.m_mu * depthFilter.m_mu ) - mu_new * mu_new;
    depthFilter.m_mu    = mu_new;
    depthFilter.m_a     = ( e - f ) / ( f - e / f );
    depthFilter.m_b     = depthFilter.m_a * ( 1.0f - f ) / f;
}

double DepthEstimator::computeTau( const Sophus::SE3d& relativePose,
                                   const Eigen::Vector3d& bearing,
                                   const double z,
                                   const double px_error_angle )
{
    const Eigen::Vector3d translation = relativePose.translation();
    const Eigen::Vector3d a           = bearing * z - translation;
    const double t_norm               = translation.norm();
    const double a_norm               = a.norm();
    const double alpha                = acos( bearing.dot( translation ) / t_norm );          // dot product
    const double beta                 = acos( a.dot( -translation ) / ( t_norm * a_norm ) );  // dot product
    const double beta_plus            = beta + px_error_angle;
    const double gamma_plus           = utils::constants::pi - alpha - beta_plus;       // triangle angles sum to PI
    const double z_plus               = t_norm * sin( beta_plus ) / sin( gamma_plus );  // law of sines
    return ( z_plus - z );                                                              // tau
}

void DepthEstimator::clearFrameQueue()
{
    // while(!m_queueFrames.empty())
    // {
    //     m_queueFrames.pop();
    // }

    // https://stackoverflow.com/a/709161/1804533
    std::queue< std::shared_ptr< Frame > > empty;
    std::swap( m_queueFrames, empty );
}

void DepthEstimator::updateFiltersLoop()
{
    while ( m_activeThread == true )
    {
        std::shared_ptr< Frame > frame;
        std::unique_lock< std::mutex > threadLocker( m_mutexFrame );
        while ( m_queueFrames.empty() && m_newKeyframeAdded == false )
            m_condition.wait( threadLocker );
        if ( m_newKeyframeAdded == true )
        {
            m_newKeyframeAdded        = false;
            m_haltUpdatingDepthFilter = false;
            clearFrameQueue();
            frame = m_newKeyframe;
        }
        else
        {
            frame = m_queueFrames.front();
            m_queueFrames.pop();
        }

        updateFilters( frame );
        if ( frame->isKeyframe() )
            initializeFilters( frame );
    }
}