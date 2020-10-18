#include "system.hpp"
#include "algorithm.hpp"
#include "utils.hpp"
#include "visualization.hpp"

#include <chrono>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>

#include "easylogging++.h"
#define System_Log( LEVEL ) CLOG( LEVEL, "System" )

// System::System( const Config& config )
System::System( const Config& config ) : m_config( &config ), m_systemStatus (System::Status::Process_First_Frame)
{
    const std::string calibrationFile = utils::findAbsoluteFilePath( m_config->m_cameraCalibrationPath );
    cv::Mat cameraMatrix;
    cv::Mat distortionCoeffs;
    loadCameraIntrinsics( calibrationFile, cameraMatrix, distortionCoeffs );
    m_camera    = std::make_shared< PinholeCamera >( m_config->m_imgWidth, m_config->m_imgHeight, cameraMatrix, distortionCoeffs );
    m_alignment = std::make_shared< ImageAlignment >( m_config->m_patchSizeImageAlignment, m_config->m_minLevelImagePyramid,
                                                      m_config->m_maxLevelImagePyramid, 6 );
    m_depthEstimator = std::make_unique <DepthEstimator> ();
    m_map = std::make_unique< Map >( m_camera, 32 );

    cv::Mat initImg (m_camera->width(), m_camera->height(), CV_8U);
    m_refFrame = std::make_shared< Frame >( m_camera, initImg, m_config->m_maxLevelImagePyramid + 1, 0.0 );

}

void System::addImage(const cv::Mat& img, const double timestamp)
{
    m_curFrame = std::make_shared< Frame >( m_camera, img, m_config->m_maxLevelImagePyramid + 1, timestamp );

    if (m_systemStatus == System::Status::Procese_New_Frame)
    {
        processNewFrame();
    }
    else if (m_systemStatus == System::Status::Process_Second_Frame)
    {
        processSecondFrame();
        return;
    }
    else if (m_systemStatus == System::Status::Process_First_Frame)
    {
        processFirstFrame();
    }
    else if (m_systemStatus == System::Status::Process_Relocalozation)
    {
        System_Log( DEBUG ) << "Relocalizations";
    }

    m_refFrame = std::move( m_curFrame );
}


void System::processFirstFrame()
{
    // m_refFrame = std::make_shared< Frame >( m_camera, firstImg, m_config->m_maxLevelImagePyramid + 1 );
    // Frame refFrame( camera, refImg );
    m_featureSelection = std::make_unique< FeatureSelection >( m_curFrame->m_imagePyramid.getBaseImage() );
    // FeatureSelection featureSelection( m_curFrame->m_imagePyramid.getBaseImage() );
    m_featureSelection->detectFeaturesInGrid( m_curFrame, m_config->m_gridPixelSize );
    // m_featureSelection->detectFeaturesByValue( m_curFrame, 150 );
    // m_featureSelection->detectFeaturesWithSSC(m_curFrame, 1000);

    // visualize
    // {
    //     cv::Mat gradient = m_featureSelection->m_gradientMagnitude.clone();
    //     cv::normalize(gradient, gradient, 0, 255, cv::NORM_MINMAX, CV_8U);
    //     cv::Mat refBGR = visualization::getBGRImage( gradient );
    //     // cv::Mat refBGR = visualization::getBGRImage( m_curFrame->m_imagePyramid.getBaseImage() );
    //     visualization::featurePoints( refBGR, m_curFrame, 5, "pink", visualization::drawingRectangle );
    //     visualization::imageGrid(refBGR, m_curFrame, m_config->m_gridPixelSize, "amber");
    //     cv::imshow("First Image", refBGR);
    //     cv::waitKey(0);
    // }

    m_curFrame->setKeyframe();
    m_map->addKeyframe( m_curFrame );
    System_Log( DEBUG ) << "Number of Features: " << m_curFrame->numberObservation();
    m_systemStatus = System::Status::Process_Second_Frame;
    // m_keyFrames.emplace_back( m_curFrame );
}

void System::processSecondFrame( )
{
    System_Log( DEBUG ) << "Number of Features: " << m_refFrame->numberObservation();

    Eigen::Matrix3d E;
    Eigen::Matrix3d F;
    Eigen::Matrix3d R;
    Eigen::Vector3d t;

    algorithm::computeOpticalFlowSparse( m_refFrame, m_curFrame, m_config->m_patchSizeOpticalFlow );
    algorithm::computeEssentialMatrix( m_refFrame, m_curFrame, 1.0, E );
    // F = m_curFrame->m_camera->invK().transpose() * E * m_refFrame->m_camera->invK();

    // decompose essential matrix to R and t and set the absolute pose of reference frame
    algorithm::recoverPose( E, m_refFrame, m_curFrame, R, t );
    System_Log( DEBUG ) << "Initial R: " << R.format( utils::eigenFormat() );
    System_Log( DEBUG ) << "Initial t: " << t.format( utils::eigenFormat() );

    // reserve the memory for depth information
    std::size_t numObserves = m_refFrame->numberObservation();
    Eigen::MatrixXd pointsWorld( 3, numObserves );
    Eigen::MatrixXd pointsCurCamera( 3, numObserves );
    Eigen::VectorXd depthCurFrame( numObserves );

    // compute the 3D points
    algorithm::triangulate3DWorldPoints( m_refFrame, m_curFrame, pointsWorld );
    algorithm::transferPointsWorldToCam( m_curFrame, pointsWorld, pointsCurCamera );
    algorithm::depthCamera( m_curFrame, pointsWorld, depthCurFrame );
    double medianDepth = algorithm::computeMedian( depthCurFrame );
    // System_Log( DEBUG ) << "Median depth in current frame: " << medianDepth;
    {
        const double minDepth = depthCurFrame.minCoeff();
        const double maxDepth = depthCurFrame.maxCoeff();
        System_Log( INFO ) << "Before SCALE, Median Depth: " << medianDepth << ", minDepth: " << minDepth << ", maxDepth: " << maxDepth;
    }

    // scale the depth
    const double scale = 1.0 / medianDepth;

    // t = -RC
    // In this formula, we assume, the W = K_{1}, means the m_refFrame->cameraInWorld() = (0, 0, 0)
    // first we compute the C, scale it and compute t
    m_curFrame->m_TransW2F.translation() =
      -m_curFrame->m_TransW2F.rotationMatrix() *
      ( m_refFrame->cameraInWorld() + scale * ( m_curFrame->cameraInWorld() - m_refFrame->cameraInWorld() ) );

    // scale current frame depth
    pointsCurCamera *= scale;
    algorithm::transferPointsCamToWorld( m_curFrame, pointsCurCamera, pointsWorld );

    uint32_t cnt = 0;
    for ( std::size_t i( 0 ); i < numObserves; i++ )
    {
        const Eigen::Vector2d refFeature = m_refFrame->m_frameFeatures[ i ]->m_feature;
        const Eigen::Vector2d curFeature = m_curFrame->m_frameFeatures[ i ]->m_feature;
        if ( m_refFrame->m_camera->isInFrame( refFeature, 5.0 ) == true && m_curFrame->m_camera->isInFrame( curFeature, 5.0 ) == true &&
             pointsCurCamera.col( i ).z() > 0 )
        {
            std::shared_ptr< Point > point = std::make_shared< Point >( pointsWorld.col( i ) );
            //    std::cout << "3D points: " << point->m_position.format(utils::eigenFormat()) << std::endl;
            m_refFrame->m_frameFeatures[ i ]->setPoint( point );
            m_curFrame->m_frameFeatures[ i ]->setPoint( point );
            point->addFeature(m_refFrame->m_frameFeatures[ i ]);
            point->addFeature(m_curFrame->m_frameFeatures[ i ]);
            cnt++;
        }
    }

    // remove the features that doesn't have the 3D point
    m_refFrame->m_frameFeatures.erase( std::remove_if( m_refFrame->m_frameFeatures.begin(), m_refFrame->m_frameFeatures.end(),
                                                       []( const auto& feature ) { return feature->m_point == nullptr; } ),
                                       m_refFrame->m_frameFeatures.end() );

    m_curFrame->m_frameFeatures.erase( std::remove_if( m_curFrame->m_frameFeatures.begin(), m_curFrame->m_frameFeatures.end(),
                                                       []( const auto& feature ) { return feature->m_point == nullptr; } ),
                                       m_curFrame->m_frameFeatures.end() );

    System_Log( INFO ) << "Init Points: " << pointsWorld.cols() << ", ref obs: " << m_refFrame->numberObservation()
                       << ", cur obs: " << m_curFrame->numberObservation();

    numObserves = m_curFrame->numberObservation();
    Eigen::VectorXd newCurDepths( numObserves );
    algorithm::depthCamera( m_curFrame, newCurDepths );
    medianDepth           = algorithm::computeMedian( newCurDepths );
    const double minDepth = newCurDepths.minCoeff();
    const double maxDepth = newCurDepths.maxCoeff();
    System_Log( INFO ) << "Before SCALE, Median Depth: " << medianDepth << ", minDepth: " << minDepth << ", maxDepth: " << maxDepth;

    // std::cout << "Mean: " << medianDepth << " min: " << minDepth << std::endl;
    m_curFrame->setKeyframe();
    System_Log( INFO ) << "Number of Features: " << m_curFrame->numberObservation();

    m_depthEstimator->addKeyframe(m_curFrame, medianDepth, 0.5 * minDepth);
    // m_keyFrames.emplace_back( m_curFrame );
    // m_depthEstimator->addKeyframe(m_curFrame, medianDepth, 0.5 * minDepth);
    m_map->addKeyframe( m_curFrame );
    m_systemStatus = System::Status::Procese_New_Frame;

    // {
    //     cv::Mat refBGR = visualization::getBGRImage( m_refFrame->m_imagePyramid.getBaseImage() );
    //     cv::Mat curBGR = visualization::getBGRImage( m_curFrame->m_imagePyramid.getBaseImage() );
    //     visualization::featurePoints( refBGR, m_refFrame, 8, "pink", visualization::drawingRectangle );
    //     // visualization::featurePointsInGrid(curBGR, curFrame, 50);
    //     // visualization::featurePoints(newBGR, newFrame);
    //     // visualization::project3DPoints(curBGR, curFrame);
    //     visualization::projectPointsWithRelativePose( curBGR, m_refFrame, m_curFrame, 8, "orange", visualization::drawingCircle );
    //     cv::Mat stickImg;
    //     visualization::stickTwoImageHorizontally( refBGR, curBGR, stickImg );
    //     cv::imshow( "both_image_1_2_optimization", stickImg );
    //     // cv::imshow("relative_1_2", curBGR);
    //     cv::waitKey( 0 );
    // }
}

void System::processNewFrame(  )
{
    // https://docs.microsoft.com/en-us/cpp/cpp/how-to-create-and-use-shared-ptr-instances?view=vs-2019
    // m_refFrame = std::move( m_curFrame );
    // std::cout << "counter ref: " << m_refFrame.use_count() << std::endl;
    // m_curFrame = std::make_shared< Frame >( m_camera, newImg, m_config->m_maxLevelImagePyramid + 1 );
    // std::cout << "counter cur: " << m_curFrame.use_count() << std::endl;

    // ImageAlignment match( m_config->m_patchSizeImageAlignment, m_config->m_minLevelImagePyramid, m_config->m_maxLevelImagePyramid, 6 );
    auto t1 = std::chrono::high_resolution_clock::now();
    m_alignment->align( m_refFrame, m_curFrame );
    auto t2 = std::chrono::high_resolution_clock::now();
    System_Log( INFO ) << "Elapsed time for alignment: " << std::chrono::duration_cast< std::chrono::microseconds >( t2 - t1 ).count()
                       << " micro sec";
    {
        cv::Mat refBGR = visualization::getBGRImage( m_refFrame->m_imagePyramid.getBaseImage() );
        cv::Mat curBGR = visualization::getBGRImage( m_curFrame->m_imagePyramid.getBaseImage() );
        visualization::featurePoints( refBGR, m_refFrame, 8, "pink", visualization::drawingRectangle );
        // visualization::featurePointsInGrid(curBGR, curFrame, 50);
        // visualization::featurePoints(newBGR, newFrame);
        // visualization::project3DPoints(curBGR, curFrame);
        visualization::projectPointsWithRelativePose( curBGR, m_refFrame, m_curFrame, 8, "orange", visualization::drawingCircle );
        cv::Mat stickImg;
        visualization::stickTwoImageHorizontally( refBGR, curBGR, stickImg );
        cv::imshow( "both_image_1_2_optimization", stickImg );
        // cv::imshow("relative_1_2", curBGR);
        cv::waitKey( 0 );
    }

    for ( const auto& refFeatures : m_refFrame->m_frameFeatures )
    {
        const auto& point      = refFeatures->m_point->m_position;
        const auto& curFeature = m_curFrame->world2image( point );
        if ( m_curFrame->m_camera->isInFrame( curFeature, 5.0 ) == true )
        {
            std::shared_ptr< Feature > newFeature = std::make_shared< Feature >( m_curFrame, curFeature, 0.0 );
            m_curFrame->addFeature( newFeature );
            m_curFrame->m_frameFeatures.back()->setPoint( refFeatures->m_point );
        }
    }
    System_Log( INFO ) << "Number of Features: " << m_curFrame->numberObservation();
    m_keyFrames.emplace_back( m_curFrame );

    // select keyframe
    // core_kfs_.insert(new_frame_);
    // setTrackingQuality(sfba_n_edges_final);
    // if(tracking_quality_ == TRACKING_INSUFFICIENT)
    // {
    //     new_frame_->T_f_w_ = last_frame_->T_f_w_; // reset to avoid crazy pose jumps
    //     return RESULT_FAILURE;
    // }
    const uint32_t numObserves = m_curFrame->numberObservation();
    Eigen::VectorXd newCurDepths( numObserves );
    algorithm::depthCamera( m_curFrame, newCurDepths );
    const double depthMean = algorithm::computeMedian( newCurDepths );
    const double depthMin  = newCurDepths.minCoeff();
    // frame_utils::getSceneDepth(*new_frame_, depth_mean, depth_min);
    // if(!needNewKf(depth_mean) || tracking_quality_ == TRACKING_BAD)
    // {
    // depth_filter_->addFrame(new_frame_);
    // return RESULT_NO_KEYFRAME;
    // }
}

void System::reportSummaryFrames()
{
    //-------------------------------------------------------------------------------
    //    Frame ID        Num Features        Num Points        Active Shared Pointer
    //-------------------------------------------------------------------------------

    std::cout << " ----------------------------- Report Summary Frames -------------------------------- " << std::endl;
    std::cout << "|                                                                                    |" << std::endl;
    for ( const auto& frame : m_keyFrames )
    {
        std::cout << "| Frame ID: " << frame->m_id << "\t\t"
                  << "Num Features: " << frame->numberObservation() << "\t\t"
                  << "Active Shared Pointer: " << frame.use_count() << "     |" << std::endl;
    }
    std::cout << "|                                                                                    |" << std::endl;
    std::cout << " ------------------------------------------------------------------------------------ " << std::endl;
}

void System::reportSummaryFeatures()
{
    std::cout << " -------------------------- Report Summary Features ---------------------------- " << std::endl;
    for ( const auto& frame : m_keyFrames )
    {
        std::cout << "|                                                                               |" << std::endl;
        std::cout << " -------------------------------- Frame ID: " << frame->m_id << " ---------------------------------- " << std::endl;
        std::cout << "|                                                                               |" << std::endl;
        for ( const auto& feature : frame->m_frameFeatures )
        {
            std::cout << "| Feature ID: " << std::left << std::setw( 12 ) << feature->m_id << "Point ID: " << std::left << std::setw( 12 )
                      << feature->m_point->m_id << "Cnt Shared Pointer Point: " << std::left << std::setw( 6 )
                      << feature->m_point.use_count() << "|" << std::endl;
        }
        std::cout << " ------------------------------------------------------------------------------- " << std::endl;
    }
}

void System::reportSummaryPoints()
{
}

void System::makeKeyframe( std::shared_ptr< Frame >& frame, const double& depthMean, const double& depthMin )
{
    frame->setKeyframe();
    //     for(Features::iterator it=new_frame_->fts_.begin(); it!=new_frame_->fts_.end(); ++it)
    //     if((*it)->point != NULL)
    //       (*it)->point->addFrameRef(*it);
    //   map_.point_candidates_.addCandidatePointToFrame(new_frame_);
    for ( const auto& feature : frame->m_frameFeatures )
    {
        if ( feature->m_point != nullptr )
        {
            // feature->m_point->
        }
    }

    // #ifdef USE_BUNDLE_ADJUSTMENT
    // if(Config::lobaNumIter() > 0)
    // {
    //     SVO_START_TIMER("local_ba");
    //     setCoreKfs(Config::coreNKfs());
    //     size_t loba_n_erredges_init, loba_n_erredges_fin;
    //     double loba_err_init, loba_err_fin;
    //     ba::localBA(new_frame_.get(), &core_kfs_, &map_,
    //                 loba_n_erredges_init, loba_n_erredges_fin,
    //                 loba_err_init, loba_err_fin);
    //     SVO_STOP_TIMER("local_ba");
    //     SVO_LOG4(loba_n_erredges_init, loba_n_erredges_fin, loba_err_init, loba_err_fin);
    //     SVO_DEBUG_STREAM("Local BA:\t RemovedEdges {"<<loba_n_erredges_init<<", "<<loba_n_erredges_fin<<"} \t "
    //                         "Error {"<<loba_err_init<<", "<<loba_err_fin<<"}");
    // }
    // #endif

    // init new depth-filters
    // depth_filter_->addKeyframe(new_frame_, depth_mean, 0.5*depth_min);
    // m_depthEstimator->addKeyframe( frame, depthMean, 0.5 * depthMin );

    // if limited number of keyframes, remove the one furthest apart
    if ( 10 > 2 && m_map->m_keyFrames.size() >= 10 )
    {
        auto futhrestFrame = m_map->getFurthestKeyframe( frame->cameraInWorld() );
        // depth_filter_->removeKeyframe(futhrestFrame); // TODO this interrupts the mapper thread, maybe we can solve this better
        m_map->removeFrame( futhrestFrame );
    }

    // add keyframe to map
    m_map->addKeyframe( frame );
}

bool System::needKeyframe( const double sceneDepthMean )
{
    return true;
}

bool System::loadCameraIntrinsics( const std::string& filename, cv::Mat& cameraMatrix, cv::Mat& distortionCoeffs )
{
    try
    {
        cv::FileStorage fs( filename, cv::FileStorage::READ );
        if ( !fs.isOpened() )
        {
            std::cout << "Failed to open " << filename << std::endl;
            return false;
        }

        fs[ "K" ] >> cameraMatrix;
        fs[ "d" ] >> distortionCoeffs;
        fs.release();
        return true;
    }
    catch ( std::exception& e )
    {
        std::cout << e.what() << std::endl;
        return false;
    }
}