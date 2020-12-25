#ifndef __SYSTEM_HPP__
#define __SYSTEM_HPP__
#include "bundle_adjustment.hpp"
#include "config.hpp"
#include "depth_estimator.hpp"
#include "feature_selection.hpp"
#include "frame.hpp"
#include "image_alignment.hpp"
#include "map.hpp"
#include "pinhole_camera.hpp"

#include <iomanip>
#include <iostream>
#include <memory>

#include <Eigen/Core>
#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>
class System final
{
    using frameSize = std::pair< const std::shared_ptr< Frame >, int32_t >;

    enum class Status : uint8_t
    {
        Process_First_Frame    = 0,
        Process_Second_Frame   = 1,
        Procese_New_Frame      = 2,
        Process_Relocalization = 3,
        Process_Default        = 4,
        Process_Paused         = 5
    };

    enum class Result : uint8_t
    {
        Success = 0,
        Failed = 1,
        Keyframe = 2
    };

public:
    explicit System( const std::shared_ptr< Config >& config );
    System( const System& rhs ) = delete;
    System( System&& rhs )      = delete;
    System& operator=( const System& rhs ) = delete;
    System& operator=( System&& rhs ) = delete;
    ~System()                         = default;

    void addImage( const cv::Mat& img, const uint64_t timestamp );

    void reportSummaryFrames();
    void reportSummaryFeatures();
    void reportSummaryPoints();

private:
    bool loadCameraIntrinsics( const std::string& filename, cv::Mat& cameraMatrix, cv::Mat& distortionCoeffs );
    bool needKeyframe( const double sceneDepthMean, const std::vector< frameSize >& overlapKeyFrames );

    Result processFirstFrame();
    Result processSecondFrame();
    Result processNewFrame();
    Result relocalizeFrame( Sophus::SE3d& pose, std::shared_ptr< Frame >& closestKeyframe );

    const std::shared_ptr< Config > m_config;
    std::shared_ptr< PinholeCamera > m_camera;
    std::shared_ptr< Frame > m_refFrame;
    std::shared_ptr< Frame > m_curFrame;
    std::shared_ptr< FeatureSelection > m_featureSelector;
    std::vector< std::shared_ptr< Frame > > m_keyFrames;
    std::unique_ptr< DepthEstimator > m_depthEstimator;
    std::unique_ptr< Map > m_map;
    std::shared_ptr< ImageAlignment > m_alignment;
    std::shared_ptr< BundleAdjustment > m_bundler;
    Status m_systemStatus;
};

#endif /* __SYSTEM_HPP__ */