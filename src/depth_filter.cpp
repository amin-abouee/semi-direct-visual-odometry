#include "depth_filter.hpp"

#include "easylogging++.h"
#define Map_Log( LEVEL ) CLOG( LEVEL, "DepthFilter" )

DepthFilter::DepthFilter()
    : m_haltUpdatingSeed( false ), m_activeThread( false ), m_newKeyframeMinDepth( 0.0 ), m_newKeyframeMeanDepth( 0.0 )
{
    // https://stackoverflow.com/a/18376082/1804533
    // https://thispointer.com/c-11-multithreading-part-1-three-different-ways-to-create-threads/
    // new std::thread(std::bind(&ThreadExample::run, this)));
    m_thread = std::make_unique< std::thread >( &DepthFilter::updateSeedsLoop, this );
}

DepthFilter::~DepthFilter()
{
    if ( m_thread != nullptr && m_thread->joinable() )
    {
        m_haltUpdatingSeed = true;
        // m_thread_->interrupt();
        m_activeThread = false;
        m_thread->join();
        // thread_ = NULL;
        m_thread = nullptr;
    }
}

void DepthFilter::addFrame( std::shared_ptr< Frame >& frame )
{
    if ( m_thread != nullptr )
    {
        std::unique_lock< std::mutex > threadLocker( m_mutexFrame );
        if ( m_queueFrames.size() > 2 )
            m_queueFrames.pop();
        m_queueFrames.push( frame );
        m_haltUpdatingSeed = false;
        m_condition.notify_one();
    }
    else
    {
        updateSeeds( frame );
    }
}

void DepthFilter::addKeyframe( std::shared_ptr< Frame >& frame, double depthMean, double depthMin )
{
    m_newKeyframeMinDepth = depthMin;
    m_newKeyframeMinDepth = depthMin;
    if ( m_thread != nullptr )
    {
        m_newKeyframe      = frame;
        m_haltUpdatingSeed = true;
        m_newKeyframeAdded = true;
        m_condition.notify_one();
    }
    else
    {
        initializeSeeds( frame );
    }
}

void DepthFilter::removeKeyframe( std::shared_ptr< Frame >& frame )
{
    m_haltUpdatingSeed = true;
    std::unique_lock< std::mutex > threadLocker(m_mutexSeed); 
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
    m_haltUpdatingSeed = false;
}

void DepthFilter::reset()
{
    m_haltUpdatingSeed = true;
    std::unique_lock< std::mutex > threadLocker(m_mutexSeed); 
    // seeds_.clear();
    //TODO: check this line
    // std::unique_lock< std::mutex > threadLocker();
    while(!m_queueFrames.empty())
        m_queueFrames.pop();
    m_haltUpdatingSeed = false;
}

double DepthFilter::computeTau( const Sophus::SE3d& T_ref_cur, const Eigen::Vector3d& f, const double z, const double px_error_angle )
{
}

void DepthFilter::initializeSeeds( std::shared_ptr< Frame >& frame )
{
    // detect new feature

    // lock mutex
    // load the feature

    // create seeds

    // set halt updating seed

    m_haltUpdatingSeed = true;
    std::unique_lock< std::mutex > threadLocker(m_mutexSeed); // by locking the updateSeeds function stops
    // ++Seed::batch_counter;
//   std::for_each(new_features.begin(), new_features.end(), [&](Feature* ftr){
//     seeds_.push_back(Seed(ftr, new_keyframe_mean_depth_, new_keyframe_min_depth_));
//   });

    m_haltUpdatingSeed = false;
}

void DepthFilter::updateSeeds( std::shared_ptr< Frame >& frame )
{
}

void DepthFilter::clearFrameQueue()
{
    // while(!m_queueFrames.empty())
    // {
    //     m_queueFrames.pop();
    // }

    // https://stackoverflow.com/a/709161/1804533
    std::queue< std::shared_ptr< Frame > > empty;
    std::swap( m_queueFrames, empty );
}

void DepthFilter::updateSeedsLoop()
{
    while ( m_activeThread == true )
    {
        std::shared_ptr< Frame > frame;
        std::unique_lock< std::mutex > threadLocker( m_mutexFrame );
        while ( m_queueFrames.empty() && m_newKeyframeAdded == false )
            m_condition.wait( threadLocker );
        if ( m_newKeyframeAdded == true )
        {
            m_newKeyframeAdded = false;
            m_haltUpdatingSeed = false;
            clearFrameQueue();
            frame = m_newKeyframe;
        }
        else
        {
            frame = m_queueFrames.front();
            m_queueFrames.pop();
        }

        updateSeeds( frame );
        if ( frame->isKeyframe() )
            initializeSeeds( frame );
    }
}