#include "feature.hpp"
#include "pinhole_camera.hpp"
#include "point.hpp"

#include <gtest/gtest.h>
#include <opencv2/imgcodecs.hpp>

using namespace testing;

class FeatureTest : public Test
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    Eigen::Vector2d pixelPosition;
    std::shared_ptr< Frame > featureFrame;
    std::shared_ptr< Camera > featureCamera;
    cv::Mat testImage         = cv::Mat( 480, 640, CV_8UC1 );
    double timestampTestImage = 1594719652204528000.0;
    uint32_t maxPyramidLevels = 4U;
    FeatureTest()
    {
        pixelPosition = Eigen::Vector2d( 10.0, 10.0 );
        cv::randu( testImage, cv::Scalar( 0 ), cv::Scalar( 255 ) );
        featureCamera = std::make_shared< PinholeCamera >( 640, 480, Eigen::Vector2d( 30.3, 40.4 ), Eigen::Vector2d( 50.5, 60.6 ) );
        featureFrame  = std::make_shared< Frame >( featureCamera, testImage, maxPyramidLevels, timestampTestImage );
    }
};

TEST_F( FeatureTest, ConstructFromLocation )
{
    uint8_t level              = 0U;
    Feature constructedFeature = Feature( featureFrame, pixelPosition, level );
    EXPECT_EQ( constructedFeature.m_frame.get(), featureFrame.get() );
    EXPECT_EQ( pixelPosition.x(), constructedFeature.m_pixelPosition.x() );
    EXPECT_EQ( pixelPosition.y(), constructedFeature.m_pixelPosition.y() );
    EXPECT_EQ( pixelPosition.x(), constructedFeature.m_homogenous.x() );
    EXPECT_EQ( pixelPosition.y(), constructedFeature.m_homogenous.y() );
    EXPECT_EQ( 1.0, constructedFeature.m_homogenous.z() );
    EXPECT_EQ( level, constructedFeature.m_level );
}

TEST_F( FeatureTest, ConstructFromFeature )
{
    double gradientMagnitude   = 1.0;
    double gradientOrientation = 1.0;
    uint8_t level              = 0U;
    Feature constructedFeature = Feature( featureFrame, pixelPosition, gradientMagnitude, gradientOrientation, level );
    EXPECT_EQ( constructedFeature.m_frame.get(), featureFrame.get() );
    EXPECT_EQ( pixelPosition.x(), constructedFeature.m_pixelPosition.x() );
    EXPECT_EQ( pixelPosition.y(), constructedFeature.m_pixelPosition.y() );
    EXPECT_EQ( pixelPosition.x(), constructedFeature.m_homogenous.x() );
    EXPECT_EQ( pixelPosition.y(), constructedFeature.m_homogenous.y() );
    EXPECT_EQ( 1.0, constructedFeature.m_homogenous.z() );
    EXPECT_EQ( gradientMagnitude, constructedFeature.m_gradientMagnitude );
    EXPECT_EQ( gradientOrientation, constructedFeature.m_gradientOrientation );
    EXPECT_EQ( level, constructedFeature.m_level );
}

TEST_F( FeatureTest, TestSetPoint )
{
    Feature constructedFeature = Feature( featureFrame, pixelPosition, 0U );
    Eigen::Vector3d point3dWCS( 10.0, 10.0, 50.0 );
    std::shared_ptr< Point > pointToSet = std::make_shared< Point >( point3dWCS );
    constructedFeature.setPoint( pointToSet );
}