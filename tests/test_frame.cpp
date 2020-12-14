#include "feature.hpp"
#include "frame.hpp"
#include "pinhole_camera.hpp"

#include <gtest/gtest.h>
#include <Eigen/Core>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <sophus/se3.hpp>

using namespace testing;

class FrameTest : public Test
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    cv::Mat testImage;
    double timestampTestImage = 1594719652204528000.0;
    uint32_t maxPyramidLevels = 4U;
    std::shared_ptr< Camera > sampleCamera;
    std::shared_ptr< Frame > sampleFrame;
    FrameTest()
    {
        testImage = cv::Mat( 480, 640, CV_8UC1 );
        cv::randu( testImage, cv::Scalar( 0 ), cv::Scalar( 255 ) );
        sampleCamera =
          std::make_shared< PinholeCamera >( 640, 480, Eigen::Vector2d( 150.0, 150.0 ), Eigen::Vector2d( 320.0, 240.0 ), nullptr );
        sampleFrame = std::make_shared< Frame >( sampleCamera, testImage, maxPyramidLevels, timestampTestImage );
    }
};

TEST_F( FrameTest, TestConstruct )
{
    Eigen::Matrix3d rotation    = Eigen::Matrix3d::Identity();
    Eigen::Vector3d translation = Eigen::Vector3d::Zero();

    std::shared_ptr<Frame> constructedFrame = std::make_shared<Frame>( sampleCamera, testImage, maxPyramidLevels, timestampTestImage );
    EXPECT_EQ( constructedFrame->m_id, 1 );
    EXPECT_EQ( constructedFrame->m_frameCounter, 2 );
    EXPECT_EQ( constructedFrame->m_absolutePose.rotationMatrix(), rotation );
    EXPECT_EQ( constructedFrame->m_absolutePose.translation(), translation );
    EXPECT_EQ( constructedFrame->m_imagePyramid.getSizeImagePyramid(), maxPyramidLevels );
}

TEST_F( FrameTest, TestSetKeyframe )
{
    EXPECT_EQ( sampleFrame->isKeyframe(), false );
    sampleFrame->setKeyframe();
    EXPECT_EQ( sampleFrame->isKeyframe(), true );
}

TEST_F( FrameTest, TestAddFeature )
{
    uint8_t level                             = 0U;
    std::shared_ptr< Feature > sampleFeature1 = std::make_shared< Feature >( sampleFrame, Eigen::Vector2d( 10.0, 10.0 ), level );
    sampleFrame->addFeature( sampleFeature1 );
    // Test size
    EXPECT_EQ( sampleFrame->m_frameFeatures.size(), 1 );
    // Test id
    EXPECT_EQ( sampleFrame->m_frameFeatures.back()->m_id, sampleFeature1->m_id );
    // Test the x coordinate of last added feature
    EXPECT_EQ( sampleFrame->m_frameFeatures.back()->m_pixelPosition.x(), sampleFeature1->m_pixelPosition.x() );
    // Test the y coordinate of last added feature
    EXPECT_EQ( sampleFrame->m_frameFeatures.back()->m_pixelPosition.y(), sampleFeature1->m_pixelPosition.y() );
    // Test to make sure the z of homogenous is 1.0
    EXPECT_EQ( sampleFrame->m_frameFeatures.back()->m_homogenous.z(), 1.0 );

    std::shared_ptr< Feature > sampleFeature2 = std::make_shared< Feature >( sampleFrame, Eigen::Vector2d( 20.0, 20.0 ), level );
    // Add the second feature
    sampleFrame->addFeature( sampleFeature2 );
    // Make sure that the size is 2
    EXPECT_EQ( sampleFrame->m_frameFeatures.size(), 2 );
    // Test the address of sampleFeature2 and the feature which was added to featurelist. Both have to refer to the same memory address
    EXPECT_EQ( sampleFrame->m_frameFeatures.back().get(), sampleFeature2.get() );
}