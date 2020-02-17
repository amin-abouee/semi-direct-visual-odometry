#ifndef __SYSTEM_HPP__
#define __SYSTEM_HPP__
#include "system.hpp"
#include "pinhole_camera.hpp"
#include "frame.hpp"
#include "feature_selection.hpp"
#include "config.hpp"

#include <iostream>
#include <memory>
#include <iomanip>
#include <opencv2/core.hpp>
#include <Eigen/Core>
#include <nlohmann/json.hpp>

class System final
{
public:
    std::shared_ptr<PinholeCamera> m_camera;
    std::shared_ptr<Frame> m_refFrame;
    std::shared_ptr<Frame> m_curFrame;
    std::unique_ptr<FeatureSelection> m_featureSelection;
    std::vector < std::shared_ptr<Frame> > m_allKeyFrames;

    explicit System(Config& config);
    System (const System& rhs) = delete;
    System (System&& rhs) = delete;
    System& operator= (const System& rhs) = delete;
    System& operator= (System&& rhs) = delete;
    ~System() = default;

    void processFirstFrame(const cv::Mat& firstImg);
    void processSecondFrame(const cv::Mat& secondImg);
    void processNewFrame(const cv::Mat& newImg);

    void reportSummaryFrames();
    void reportSummaryFeatures();
    void reportSummaryPoints();

private:
    bool loadCameraIntrinsics( const std::string& filename, cv::Mat& cameraMatrix, cv::Mat& distortionCoeffs );

    Config* m_config;
};

#endif /* __SYSTEM_HPP__ */