#ifndef __SYSTEM_HPP__
#define __SYSTEM_HPP__
#include "config.hpp"
#include "depth_estimator.hpp"
#include "feature_selection.hpp"
#include "frame.hpp"
#include "image_alignment.hpp"
#include "map.hpp"
#include "pinhole_camera.hpp"
#include "system.hpp"

#include <iomanip>
#include <iostream>
#include <memory>

#include <Eigen/Core>
#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>

class System final
{
    enum class Status : uint8_t
    {
        Process_First_Frame    = 0,
        Process_Second_Frame   = 1,
        Procese_New_Frame      = 2,
        Process_Relocalozation = 3,
        Process_Default        = 4,
        Process_Paused         = 5
    };

public:
    std::shared_ptr< PinholeCamera > m_camera;
    std::shared_ptr< Frame > m_refFrame;
    std::shared_ptr< Frame > m_curFrame;
    std::unique_ptr< FeatureSelection > m_featureSelection;
    std::vector< std::shared_ptr< Frame > > m_keyFrames;
    std::unique_ptr< DepthEstimator > m_depthEstimator;
    std::unique_ptr< Map > m_map;
    Status m_systemStatus;

    explicit System( const Config& config );
    System( const System& rhs ) = delete;
    System( System&& rhs )      = delete;
    System& operator=( const System& rhs ) = delete;
    System& operator=( System&& rhs ) = delete;
    ~System()                         = default;

    void addImage( const cv::Mat& img, const double timestamp );

    void processFirstFrame( );
    void processSecondFrame( );
    void processNewFrame( );

    void reportSummaryFrames();
    void reportSummaryFeatures();
    void reportSummaryPoints();

private:
    bool loadCameraIntrinsics( const std::string& filename, cv::Mat& cameraMatrix, cv::Mat& distortionCoeffs );
    bool needKeyframe( const double sceneDepthMean );
    void makeKeyframe( std::shared_ptr< Frame >& frame, const double& depthMean, const double& depthMin );

    std::shared_ptr< ImageAlignment > m_alignment;
    const Config* m_config;
};

#endif /* __SYSTEM_HPP__ */