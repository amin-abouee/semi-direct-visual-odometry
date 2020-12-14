#include "feature.hpp"
#include "feature_selection.hpp"
#include "pinhole_camera.hpp"
#include "point.hpp"
#include "visualization.hpp"

#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <Eigen/Core>

#include "easylogging.h"

using namespace testing;

class FeatureSelectionTest : public Test
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    double timestampTestImage = 1594719652204528000.0;
    uint32_t maxPyramidLevels = 4U;
    std::shared_ptr<Camera> sampleCamera;
    std::shared_ptr<Frame> sampleFrame;

    cv::Mat imageColor;
    cv::Mat image;

    const float detectionThreshold {50.0f};
    const int32_t cellSize {50};

    FeatureSelectionTest()
    {
        //testImage = cv::Mat(640, 480, CV_8UC1);
        //cv::randu(testImage, cv::Scalar(0), cv::Scalar(255));
        image = cv::imread(TEST_DATA_DIR"images/image_2.jpg", cv::IMREAD_GRAYSCALE);
        imageColor = cv::imread(TEST_DATA_DIR"images/image_2.jpg", cv::IMREAD_COLOR);
        sampleCamera = std::make_shared<PinholeCamera>(1920, 1080, Eigen::Vector2d(150.0, 150.0), Eigen::Vector2d(320.0, 240.0), nullptr);
        sampleFrame = std::make_shared<Frame>(sampleCamera, image, maxPyramidLevels, timestampTestImage);
    }

private: 

};

TEST_F(FeatureSelectionTest, ConstructFromLocation)
{
    // Constructor
    std::shared_ptr<FeatureSelection> featureSelector = std::make_shared<FeatureSelection>(1920, 1080, 150);
}


TEST_F(FeatureSelectionTest, TestGradientMagnitudeWithSSCGrid)
{
    std::shared_ptr<Frame> frame = std::make_shared<Frame>(sampleCamera, image, maxPyramidLevels, timestampTestImage);
    std::shared_ptr<FeatureSelection> featureSelector = std::make_shared<FeatureSelection>(1920, 1080, cellSize);

    {
        TIMED_SCOPE(timerSSCGrid, "gradient_magnitude_ssc_grid");
        featureSelector->gradientMagnitudeWithSSC(frame, detectionThreshold, 500, true);
    }

    cv::Mat imageTest = image.clone();
    imageTest = visualization::getColorImage(imageTest);
    visualization::imageGrid(imageTest, cellSize, "amber", 0);
    visualization::featurePoints(imageTest, frame, 4.0, "pink");

    cv::imwrite("TestGradientMagnitudeWithSSC_grid.png", imageTest);
}

TEST_F(FeatureSelectionTest, TestGradientMagnitudeWithSSCNoGrid)
{
    std::shared_ptr<Frame> frame = std::make_shared<Frame>(sampleCamera, image, maxPyramidLevels, timestampTestImage);
    std::shared_ptr<FeatureSelection> featureSelector = std::make_shared<FeatureSelection>(1920, 1080, cellSize);

    {
        TIMED_SCOPE(timerSSCNoGrid, "gradient_magnitude_ssc_no_grid");
        featureSelector->gradientMagnitudeWithSSC(frame, detectionThreshold, 500, false);
    }

    cv::Mat imageTest = image.clone();
    imageTest = visualization::getColorImage(imageTest);
    visualization::featurePoints(imageTest, frame, 4.0, "pink");

    cv::imwrite("TestGradientMagnitudeWithSSC_no_grid.png", imageTest);
}

TEST_F(FeatureSelectionTest, TestGradientMagnitudeByValueGrid)
{
    std::shared_ptr<Frame> frame = std::make_shared<Frame>(sampleCamera, image, maxPyramidLevels, timestampTestImage);
    std::shared_ptr<FeatureSelection> featureSelector = std::make_shared<FeatureSelection>(1920, 1080, cellSize);

    {
        TIMED_SCOPE(timerValueGrid, "gradient_magnitude_value_grid");
        featureSelector->gradientMagnitudeByValue(frame, detectionThreshold, true);
    }

    cv::Mat imageTest = image.clone();
    imageTest = visualization::getColorImage(imageTest);
    visualization::imageGrid(imageTest, cellSize, "amber", 0);
    visualization::featurePoints(imageTest, frame, 4.0, "pink");

    cv::imwrite("TestGradientMagnitudeByValue_grid.png", imageTest);
}

TEST_F(FeatureSelectionTest, TestGradientMagnitudeByValueNoGrid)
{
    std::shared_ptr<Frame> frame = std::make_shared<Frame>(sampleCamera, image, maxPyramidLevels, timestampTestImage);
    std::shared_ptr<FeatureSelection> featureSelector = std::make_shared<FeatureSelection>(1920, 1080, cellSize);

    {
        TIMED_SCOPE(timerValueNoGrid, "gradient_magnitude_value_no_grid");
        featureSelector->gradientMagnitudeByValue(frame, detectionThreshold, false);
    }

    cv::Mat imageTest = image.clone();
    imageTest = visualization::getColorImage(imageTest);
    // visualization::imageGrid(imageTest, cellSize, "amber", 0);
    visualization::featurePoints(imageTest, frame, 4.0, "pink");

    cv::imwrite("TestGradientMagnitudeByValue_no_grid.png", imageTest);
}

TEST_F(FeatureSelectionTest, TestKFastAndOctreeGrid)
{
    std::shared_ptr<Frame> frame = std::make_shared<Frame>(sampleCamera, image, maxPyramidLevels, timestampTestImage);
    std::shared_ptr<FeatureSelection> featureSelector = std::make_shared<FeatureSelection>(1920, 1080, cellSize);

    {
        TIMED_SCOPE(timerKFastNoGrid, "gradient_magnitude_kfast_grid");
        featureSelector->KFastAndOctreeDetector(frame, 10, 5000, true);
    }

    cv::Mat imageTest = image.clone();
    imageTest = visualization::getColorImage(imageTest);
    visualization::imageGrid(imageTest, cellSize, "amber", 0);
    visualization::featurePoints(imageTest, frame, 4.0, "pink");

    cv::imwrite("TestKFastAndOctree_grid.png", imageTest);
}

TEST_F(FeatureSelectionTest, TestKFastAndOctreeNoGrid)
{
    std::shared_ptr<Frame> frame = std::make_shared<Frame>(sampleCamera, image, maxPyramidLevels, timestampTestImage);
    std::shared_ptr<FeatureSelection> featureSelector = std::make_shared<FeatureSelection>(1920, 1080, cellSize);

    {
        TIMED_SCOPE(timerKFastNoGrid, "gradient_magnitude_kfast_no_grid");
        featureSelector->KFastAndOctreeDetector(frame, 10, 5000, false);
    }

    cv::Mat imageTest = image.clone();
    imageTest = visualization::getColorImage(imageTest);
    // visualization::imageGrid(imageTest, cellSize, "amber", 0);
    visualization::featurePoints(imageTest, frame, 4.0, "pink");

    cv::imwrite("TestKFastAndOctree_no_grid.png", imageTest);
}