#include "depth_estimator.hpp"
#include "algorithm.hpp"
#include "utils.hpp"
#include "visualization.hpp"

#include <opencv2/highgui.hpp>
#include <easylogging++.h>

#include <random>

#define Depth_Log( LEVEL ) CLOG( LEVEL, "Depth" )

DepthEstimator::DepthEstimator( std::shared_ptr< FeatureSelection >& featureSelection )
    : m_deletedKeyframe( nullptr )
    , m_featureSelector( featureSelection )
    , m_newKeyframeMinDepth( 0.0 )
    , m_newKeyframeMeanDepth( 0.0 )
    , m_haltUpdatingDepthFilter( false )
    , m_newKeyframeAdded( false )
    , m_activeThread( true )
    , m_terminateThread( false )
{
    // https://stackoverflow.com/a/18376082/1804533
    // https://thispointer.com/c-11-multithreading-part-1-three-different-ways-to-create-threads/
    // https://stackoverflow.com/a/10673671
    // new std::thread(std::bind(&ThreadExample::run, this)));
    // https://www.youtube.com/watch?v=hCvc9y39RDw&list=PLk6CEY9XxSIAeK-EAh3hB4fgNvYkYmghp&index=2
    m_thread = std::make_unique< std::thread >( &DepthEstimator::updateFiltersLoop, this );
    // Depth_Log( DEBUG ) << "thread initiliazed";
    // Depth_Log( DEBUG ) << "newKeyframeMinDepth: " << m_newKeyframeMinDepth << ", newKeyframeMinDepth: " << m_newKeyframeMinDepth;
}

DepthEstimator::~DepthEstimator()
{
    Depth_Log( DEBUG ) << "D'tor DepthEstimation ";
    m_terminateThread = true;
    m_condition.notify_one();
    while ( m_thread != nullptr && m_thread->joinable() )
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
    m_newKeyframeMeanDepth = depthMean;
    m_newKeyframeMinDepth  = depthMin;

    Depth_Log( DEBUG ) << "newKeyframeMinDepth: " << m_newKeyframeMinDepth << ", newKeyframeMeanDepth: " << m_newKeyframeMeanDepth;

    if ( m_thread != nullptr )
    {
        std::unique_lock< std::mutex > threadLocker( m_mutexFrame );
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
    m_deletedKeyframe = frame;
    // m_haltUpdatingDepthFilter = true;
    // std::unique_lock< std::mutex > threadLocker( m_mutexFilter );
    // auto element = std::remove_if( m_depthFilters.begin(), m_depthFilters.end(), [ &frame ]( auto& depthFilter ) -> bool {
    //     if ( depthFilter.m_feature->m_frame == frame )
    //         return true;
    //     return false;
    // } );
    // m_depthFilters.erase( element, m_depthFilters.end() );

    // m_haltUpdatingDepthFilter = false;
}

void DepthEstimator::reset()
{
    m_haltUpdatingDepthFilter = true;
    std::unique_lock< std::mutex > threadLocker( m_mutexFilter );
    m_depthFilters.clear();
    // seeds_.clear();
    // TODO: check this line
    // std::unique_lock< std::mutex > threadLocker();
    while ( !m_queueFrames.empty() )
        m_queueFrames.pop();
    m_haltUpdatingDepthFilter = false;
}

void DepthEstimator::updateFiltersLoop()
{
    // TODO: change the thread name of depth estimation
    // el::Logger* l = el::Loggers::getLogger("Depth");
    // el::Helpers::setThreadName("depth-thread");

    while ( m_activeThread == true )
    {
        std::shared_ptr< Frame > frame;
        std::unique_lock< std::mutex > threadLocker( m_mutexFrame );
        while ( m_queueFrames.empty() && m_newKeyframeAdded == false && m_terminateThread == false )
        {
            m_condition.wait( threadLocker );
        }

        if ( m_terminateThread == true )
            return;
        // while ( m_queueFrames.empty() && m_newKeyframeAdded == false )
        // m_condition.wait( threadLocker , [this] {return m_queueFrames.empty() && m_newKeyframeAdded == false;} );

        if ( m_newKeyframeAdded == true )
        {
            Depth_Log( INFO ) << "New key frame " << m_newKeyframe->m_id << " added to depth estimation";
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

        if ( m_deletedKeyframe != nullptr )
        {
            removeKeyframe();
        }

        updateFilters( frame );
        Depth_Log( INFO ) << "Depth of frame " << frame->m_id << " updated.";

        if ( frame->isKeyframe() )
        {
            initializeFilters( frame );
            Depth_Log( INFO ) << "Depth of key frame " << frame->m_id << " initialized.";
        }
    }
}

void DepthEstimator::removeKeyframe()
{
    std::unique_lock< std::mutex > threadLocker( m_mutexFilter );
    auto element = std::remove_if( m_depthFilters.begin(), m_depthFilters.end(), [this]( auto& depthFilter ) -> bool {
        if ( depthFilter.m_feature->m_frame == m_deletedKeyframe )
            return true;
        return false;
    } );
    m_depthFilters.erase( element, m_depthFilters.end() );
    m_deletedKeyframe = nullptr;
}


void DepthEstimator::initializeFilters( std::shared_ptr< Frame >& frame )
{
    Depth_Log( DEBUG ) << "initializeFilters";

    // m_haltUpdatingDepthFilter = true;
    // std::unique_lock< std::mutex > threadLocker( m_mutexFilter );

    for ( auto& feature : frame->m_features )
    {
        m_depthFilters.push_back( MixedGaussianFilter( feature, m_newKeyframeMeanDepth, m_newKeyframeMinDepth ) );
    }
    // m_haltUpdatingDepthFilter = false;

    Depth_Log( DEBUG ) << m_depthFilters.size() << " filters initialized for depth estimation";

    // if (m_depthFilters.size() > 0)
    // {
    //     cv::Mat refBGR = visualization::getColorImageImage( m_depthFilters[0].m_feature->m_frame->m_imagePyramid.getBaseImage() );
    //     visualization::projectDepthFilters(refBGR, frame, m_depthFilters, 4, "lime", visualization::drawingLine);
    //     std::stringstream ss;
    //     ss << "create depth for frame " << frame->m_id;
    //     cv::imshow( ss.str(), refBGR );
    //     // cv::imshow( "create depth", refBGR );
    //     cv::waitKey(0);
    // }
}

void DepthEstimator::updateFilters( std::shared_ptr< Frame >& frame )
{
    // update only a limited number of seeds, because we don't have time to do it
    // for all the seeds in every frame!
    uint32_t successUpdated = 0.0;
    uint32_t failedUpdated  = 0.0;
    Depth_Log( DEBUG ) << "updateFilters";
    Depth_Log( DEBUG ) << "m_haltUpdatingDepthFilter: " << m_haltUpdatingDepthFilter;

    std::unique_lock< std::mutex > threadLocker( m_mutexFilter );  // by locking the updateSeeds function stops
    const double focalLength = frame->m_camera->fx();
    const double pixelNoise  = 1.0;
    // tan (theta/2) = W/(2*f) eq. 2.60, computer vision, algorithms and applications
    // here, W = 1 (pixel error). theta = arctan(1/2*f)*2
    const double pixelErrorAngle = atan( pixelNoise / ( 2.0 * focalLength ) ) * 2.0;  // law of chord

    Depth_Log( DEBUG ) << "Frame: " << frame->m_id << ", size of depthFilters " << m_depthFilters.size();

    // if ( m_depthFilters.size() > 0 )
    // {
    //     cv::Mat refBGR = visualization::getColorImageImage( frame->m_imagePyramid.getBaseImage() );
    //     // cv::Mat curBGR = visualization::getColorImageImage( frame->m_imagePyramid.getBaseImage() );
    //     // cv::Mat stickImg;
    //     // visualization::stickTwoImageHorizontally( refBGR, curBGR, stickImg );
    //     visualization::projectDepthFilters( refBGR, frame, m_depthFilters, 4, "lime", visualization::drawingLine );
    //     std::stringstream ss;
    //     ss << "depth for frame " << frame->m_id;
    //     cv::imshow( ss.str(), refBGR );
    //     // cv::waitKey(0);
    // }

    std::vector< double > updatedDepths;

    for ( auto& depthFilter : m_depthFilters )
    {
        Depth_Log( DEBUG ) << "depth filter " << depthFilter.m_id << ", mu: " << depthFilter.m_frameId << ", sigma: " << depthFilter.m_sigma;

        if ( m_haltUpdatingDepthFilter == true )
            return;
        // if the current depth filter is very old, delete it
        if ( MixedGaussianFilter::m_frameCounter - depthFilter.m_frameCounter > 2 )
        {
            depthFilter.m_validity = false;
            failedUpdated++;
            updatedDepths.push_back( 0.0 );
            continue;
        }

        // TODO: double check the code
        const Sophus::SE3d relativePose        = algorithm::computeRelativePose( depthFilter.m_feature->m_frame, frame );
        const Eigen::Vector3d pointInCurCamera = relativePose * ( depthFilter.m_feature->m_bearingVec / depthFilter.m_mu );
        if ( pointInCurCamera.z() < 0 || frame->m_camera->isInFrame( frame->camera2image( pointInCurCamera ) ) == false )
        {
            depthFilter.m_validity = false;
            failedUpdated++;
            updatedDepths.push_back( 0.0 );
            continue;
        }

        // inverse representation of depth
        const double inverseMinDepth = depthFilter.m_mu + depthFilter.m_var;
        const double inverseMaxDepth = std::max( depthFilter.m_mu - depthFilter.m_var, 1e-7 );
        double updatedDepth          = 0.0;

        // TODO:check in epipolar distance
        algorithm::matchEpipolarConstraint( depthFilter.m_feature->m_frame, frame, depthFilter.m_feature, 7, 1.0 / depthFilter.m_mu,
                                            1.0 / inverseMinDepth, 1.0 / inverseMaxDepth, updatedDepth );

        updatedDepths.push_back( updatedDepth );

        // compute tau
        const double tau        = computeTau( relativePose, depthFilter.m_feature->m_bearingVec, updatedDepth, pixelErrorAngle );
        const double inverseTau = 0.5 * ( 1.0 / std::max( 1e-7, updatedDepth - tau ) - 1.0 / ( updatedDepth + tau ) );

        // update the estimate
        updateFilter( 1.0 / updatedDepth, inverseTau * inverseTau, depthFilter );
        successUpdated++;

        if ( frame->isKeyframe() )
        {
            // The feature detector should not initialize new seeds close to this location
            const Eigen::Vector2d newLocation =
              frame->camera2image( relativePose * frame->image2camera( depthFilter.m_feature->m_pixelPosition, 1.0 / depthFilter.m_mu ) );
            // TODO: we need to set the corresponding location in our grid
            m_featureSelector->setCellInGridOccupancy( newLocation );
        }

        // if the filter has converged, we initialize a new candidate point and remove the seed
        if ( sqrt( depthFilter.m_var ) * 10.0 < depthFilter.m_maxDepth )
        {
            // assert( it->ftr->point == NULL );  // TODO this should not happen anymore
            const Eigen::Vector3d pointInWorld =
              depthFilter.m_feature->m_frame->image2world( depthFilter.m_feature->m_pixelPosition, 1.0 / depthFilter.m_mu );
            auto point                     = std::make_shared< Point >( pointInWorld, depthFilter.m_feature );
            depthFilter.m_feature->m_point = point;

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
            Depth_Log( WARNING ) << "Inverse depth is nan";
            depthFilter.m_validity = false;
            failedUpdated++;
        }
        Depth_Log( DEBUG ) << "updated depth filter " << depthFilter.m_id << ", mu: " << depthFilter.m_frameId << ", sigma: " << depthFilter.m_sigma;
        // Depth_Log( DEBUG ) << "Filter id: " << depthFilter.m_id << ", mu: " << depthFilter.m_mu << ", sigma: " << depthFilter.m_sigma;
    }

    // if ( m_depthFilters.size() > 0 )
    // {
    //     cv::Mat refBGR = visualization::getColorImageImage( m_depthFilters[ 0 ].m_feature->m_frame->m_imagePyramid.getBaseImage() );
    //     visualization::featurePoints( refBGR, m_depthFilters[ 0 ].m_feature->m_frame, 8, "pink", visualization::drawingRectangle );
    //     cv::Mat curBGR = visualization::getColorImageImage( frame->m_imagePyramid.getBaseImage() );
    //     visualization::projectDepthFilters( curBGR, frame, m_depthFilters, updatedDepths, 4, "lime", visualization::drawingLine );
    //     cv::Mat stickImg;
    //     visualization::stickTwoImageHorizontally( refBGR, curBGR, stickImg );
    //     std::stringstream ss;
    //     ss << "updated depth for frame " << frame->m_id;
    //     cv::imshow( ss.str(), stickImg );
    //     cv::waitKey( 0 );
    // }

    // remove bad filters
    auto removedElements = std::remove_if( m_depthFilters.begin(), m_depthFilters.end(), []( auto& depthFilter ) -> bool {
        if ( depthFilter.m_validity == false )
            return true;
        return false;
    } );
    m_depthFilters.erase( removedElements, m_depthFilters.end() );
}

void DepthEstimator::updateFilter( const double x, const double tau2, MixedGaussianFilter& depthFilter )
{
    const double norm_scale = sqrt( depthFilter.m_var + tau2 );
    if ( std::isnan( norm_scale ) )
        return;
    // std::normal_distribution< double > normalDist( depthFilter.m_mu, norm_scale );
    const double s2 = 1.0 / ( 1.0 / depthFilter.m_var + 1.0 / tau2 );
    const double m  = s2 * ( depthFilter.m_mu / depthFilter.m_var + x / tau2 );
    // https://stackoverflow.com/questions/10847007/using-the-gaussian-probability-density-function-in-c
    double C1 =
      depthFilter.m_a / ( depthFilter.m_a + depthFilter.m_b ) * algorithm::computeNormalDistribution( depthFilter.m_mu, norm_scale, x );
    double C2                           = depthFilter.m_b / ( depthFilter.m_a + depthFilter.m_b ) * 1.00 / depthFilter.m_maxDepth;
    const double normalization_constant = C1 + C2;
    C1 /= normalization_constant;
    C2 /= normalization_constant;
    const double f = C1 * ( depthFilter.m_a + 1.0 ) / ( depthFilter.m_a + depthFilter.m_b + 1.0 ) +
                     C2 * depthFilter.m_a / ( depthFilter.m_a + depthFilter.m_b + 1.0 );
    const double e = C1 * ( depthFilter.m_a + 1.0 ) * ( depthFilter.m_a + 2.0 ) /
                       ( ( depthFilter.m_a + depthFilter.m_b + 1.0 ) * ( depthFilter.m_a + depthFilter.m_b + 2.0 ) ) +
                     C2 * depthFilter.m_a * ( depthFilter.m_a + 1.0 ) /
                       ( ( depthFilter.m_a + depthFilter.m_b + 1.0 ) * ( depthFilter.m_a + depthFilter.m_b + 2.0 ) );

    // update parameters
    const double newMu  = C1 * m + C2 * depthFilter.m_mu;
    depthFilter.m_var   = C1 * ( s2 + m * m ) + C2 * ( depthFilter.m_var + depthFilter.m_mu * depthFilter.m_mu ) - newMu * newMu;
    depthFilter.m_sigma = sqrt( depthFilter.m_var );
    depthFilter.m_mu    = newMu;
    depthFilter.m_a     = ( e - f ) / ( f - e / f );
    depthFilter.m_b     = depthFilter.m_a * ( 1.0 - f ) / f;
}

double DepthEstimator::computeTau( const Sophus::SE3d& relativePose,
                                   const Eigen::Vector3d& bearing,
                                   const double depth,
                                   const double pixelErrorAngle )
{
    const Eigen::Vector3d translation = relativePose.translation();
    const Eigen::Vector3d diff        = bearing * depth - translation;
    const double normTranslation      = translation.norm();
    const double normDiff             = diff.norm();
    const double alpha                = std::acos( bearing.dot( translation ) / normTranslation );               // dot product
    const double beta                 = std::acos( diff.dot( -translation ) / ( normTranslation * normDiff ) );  // dot product
    const double betaUpdated          = beta + pixelErrorAngle;
    const double gammaUpdated         = utils::constants::pi - alpha - betaUpdated;                            // triangle angles sum to PI
    const double depthUpdated         = normTranslation * std::sin( betaUpdated ) / std::sin( gammaUpdated );  // law of sines
    return ( depthUpdated - depth );                                                                           // tau
}

void DepthEstimator::clearFrameQueue()
{
    // https://stackoverflow.com/a/709161/1804533
    std::queue< std::shared_ptr< Frame > > empty;
    std::swap( m_queueFrames, empty );
}