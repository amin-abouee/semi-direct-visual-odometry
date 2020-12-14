#include "feature.hpp"
#include "frame.hpp"
#include "pinhole_camera.hpp"
#include "point.hpp"

#include <gtest/gtest.h>
#include <Eigen/Core>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <sophus/se3.hpp>

using namespace testing;

class PointTest : public Test
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    cv::Mat testImage;
    double timestampTestImage = 1594719652204528000.0;
    uint32_t maxPyramidLevels = 4U;
    std::shared_ptr< Camera > sampleCamera;
    Eigen::Vector2d principlePoint{ 320.0, 240.0 };
    Eigen::Vector2d focalLength{ 150.0, 150.0 };
    PointTest()
    {
        testImage = cv::Mat( 480, 640, CV_8UC1 );
        cv::randu( testImage, cv::Scalar( 0 ), cv::Scalar( 255 ) );
        sampleCamera = std::make_shared< PinholeCamera >( 640, 480, focalLength, principlePoint, nullptr );
    }
};

TEST_F( PointTest, TestConstructPoint )
{
    Eigen::Vector3d point( 10.0, 10.0, 50.0 );
    std::shared_ptr< Point > samplePoint = std::make_shared< Point >( point );
    EXPECT_EQ( samplePoint->m_id, 0 );
    EXPECT_EQ( samplePoint->m_pointCounter, 1 );
    EXPECT_EQ( samplePoint->m_position, point );
    EXPECT_EQ( samplePoint->m_type, Point::PointType::UNKNOWN );
    EXPECT_EQ( samplePoint->m_features.size(), 0 );
    EXPECT_EQ( samplePoint->isNormalEstimated(), false );
}

TEST_F( PointTest, TestConstructPointAndFeature )
{
    std::shared_ptr< Frame > sampleFrame     = std::make_shared< Frame >( sampleCamera, testImage, maxPyramidLevels, timestampTestImage );
    uint8_t level                            = 0U;
    std::shared_ptr< Feature > sampleFeature = std::make_shared< Feature >( sampleFrame, Eigen::Vector2d( 5.0, 5.0 ), level );

    Eigen::Vector3d point( 10.0, 10.0, 50.0 );
    std::shared_ptr< Point > samplePoint = std::make_shared< Point >( point, sampleFeature );

    EXPECT_EQ( samplePoint->m_id, 1 );
    EXPECT_EQ( samplePoint->m_pointCounter, 2 );
    EXPECT_EQ( samplePoint->m_position, point );
    EXPECT_EQ( samplePoint->m_type, Point::PointType::UNKNOWN );
    EXPECT_EQ( samplePoint->m_features.size(), 1 );
    EXPECT_EQ( samplePoint->isNormalEstimated(), false );
    EXPECT_EQ( samplePoint->m_features.back().get(), sampleFeature.get() );
}

TEST_F( PointTest, TestAddFeature )
{
    std::shared_ptr< Frame > sampleFrame = std::make_shared< Frame >( sampleCamera, testImage, maxPyramidLevels, timestampTestImage );

    uint8_t level                             = 0U;
    std::shared_ptr< Feature > sampleFeature1 = std::make_shared< Feature >( sampleFrame, Eigen::Vector2d( 5.0, 5.0 ), level );
    std::shared_ptr< Feature > sampleFeature2 = std::make_shared< Feature >( sampleFrame, Eigen::Vector2d( 10.0, 15.0 ), level );
    std::shared_ptr< Feature > sampleFeature3 = std::make_shared< Feature >( sampleFrame, Eigen::Vector2d( 15.0, 30.0 ), level );

    Eigen::Vector3d point( 10.0, 10.0, 50.0 );
    std::shared_ptr< Point > samplePoint = std::make_shared< Point >( point );
    samplePoint->addFeature( sampleFeature1 );
    samplePoint->addFeature( sampleFeature2 );
    samplePoint->addFeature( sampleFeature3 );

    EXPECT_EQ( samplePoint->m_features.size(), 3 );
    EXPECT_EQ( samplePoint->m_features[ 1 ].get(), sampleFeature2.get() );
}

TEST_F( PointTest, TestRemoveFeature )
{
    std::shared_ptr< Frame > sampleFrame = std::make_shared< Frame >( sampleCamera, testImage, maxPyramidLevels, timestampTestImage );

    // All features from the same frame
    uint8_t level                             = 0U;
    std::shared_ptr< Feature > sampleFeature1 = std::make_shared< Feature >( sampleFrame, Eigen::Vector2d( 5.0, 5.0 ), level );
    std::shared_ptr< Feature > sampleFeature2 = std::make_shared< Feature >( sampleFrame, Eigen::Vector2d( 10.0, 15.0 ), level );
    std::shared_ptr< Feature > sampleFeature3 = std::make_shared< Feature >( sampleFrame, Eigen::Vector2d( 15.0, 30.0 ), level );

    // Three different features, but from same frame
    Eigen::Vector3d point( 10.0, 10.0, 50.0 );
    std::shared_ptr< Point > samplePoint = std::make_shared< Point >( point );
    samplePoint->addFeature( sampleFeature1 );
    samplePoint->addFeature( sampleFeature2 );
    samplePoint->addFeature( sampleFeature3 );

    EXPECT_EQ( samplePoint->m_features.size(), 3 );
    // remove all visible features from sampleFrame
    samplePoint->removeFrame( sampleFrame );
    EXPECT_EQ( samplePoint->m_features.size(), 0 );
    EXPECT_EQ( samplePoint->numberObservation(), 0 );

    // Create new frame
    std::shared_ptr< Frame > sampleFrame1     = std::make_shared< Frame >( sampleCamera, testImage, maxPyramidLevels, timestampTestImage );
    std::shared_ptr< Feature > sampleFeature4 = std::make_shared< Feature >( sampleFrame1, Eigen::Vector2d( 20.0, 20.0 ), level );

    // Add two features, but from two frames
    samplePoint->addFeature( sampleFeature1 );
    samplePoint->addFeature( sampleFeature4 );

    EXPECT_EQ( samplePoint->m_features.size(), 2 );
    // Remove the all features of sampleFrame1, we still have one feature from sampleFrame
    samplePoint->removeFrame( sampleFrame1 );

    EXPECT_EQ( samplePoint->m_features.size(), 1 );
    EXPECT_EQ( samplePoint->m_features.back()->m_frame.get(), sampleFrame.get() );
    EXPECT_EQ( samplePoint->m_features.back().get(), sampleFeature1.get() );
}

TEST_F( PointTest, TestComputeNormal )
{
    // sample rotation from world to camera coordinate
    Eigen::Matrix3d rotation;
    rotation << -1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0;

    uint8_t level = 0U;
    // Create first frame
    std::shared_ptr< Frame > sampleFrame1 = std::make_shared< Frame >( sampleCamera, testImage, maxPyramidLevels, timestampTestImage );
    sampleFrame1->m_absolutePose          = Sophus::SE3d( rotation, Eigen::Vector3d( 0.0, 0.0, 5.0 ) );
    // feature lies on the principle ray
    std::shared_ptr< Feature > sampleFeature1 = std::make_shared< Feature >( sampleFrame1, principlePoint, level );

    // Create second frame
    std::shared_ptr< Frame > sampleFrame2 = std::make_shared< Frame >( sampleCamera, testImage, maxPyramidLevels, timestampTestImage );
    sampleFrame2->m_absolutePose          = Sophus::SE3d( rotation, Eigen::Vector3d( 5.0, 0.0, 0.0 ) );
    // feature lies on the principle ray
    std::shared_ptr< Feature > sampleFeature2 = std::make_shared< Feature >( sampleFrame2, principlePoint, level );

    Eigen::Vector3d point( 0.0, 0.0, 0.0 );
    std::shared_ptr< Point > samplePoint = std::make_shared< Point >( point );
    samplePoint->addFeature( sampleFeature1 );

    // I expected to see (0.0, -1.0, 0.0)
    Eigen::Vector3d expectedNormal(0.0, -1.0, 0.0);
    // A bigger epsilon compare to numeric_limit<double>::epsilon()
    constexpr double epsilon = 1e-10;

    samplePoint->computeNormal();
    EXPECT_EQ( samplePoint->isNormalEstimated(), true );

    // first test
    // compute the dot product to estimate the angle betwen them. if both are same, the angle between them are 0° and cos(0°) = 1
    EXPECT_EQ( samplePoint->m_normal.dot( Eigen::Vector3d( 0.0, -1.0, 0.0 ) ), 1 );

    // second test
    EXPECT_NEAR( samplePoint->m_normal.x(), expectedNormal.x(), epsilon );
    EXPECT_NEAR( samplePoint->m_normal.y(), expectedNormal.y(), epsilon );
    EXPECT_NEAR( samplePoint->m_normal.z(), expectedNormal.z(), epsilon );

    // new feature added but I expected to see the same normal (0.0, -1.0, 0.0)
    samplePoint->addFeature( sampleFeature2 );
    samplePoint->computeNormal();

    // first test
    EXPECT_EQ( samplePoint->m_normal.dot( Eigen::Vector3d( 0.0, -1.0, 0.0 ) ), 1 );

    // second test
    EXPECT_NEAR( samplePoint->m_normal.x(), expectedNormal.x(), epsilon );
    EXPECT_NEAR( samplePoint->m_normal.y(), expectedNormal.y(), epsilon );
    EXPECT_NEAR( samplePoint->m_normal.z(), expectedNormal.z(), epsilon );
}

TEST_F( PointTest, TestNumberObservation )
{
    uint8_t level = 0U;
    // Create first frame
    std::shared_ptr< Frame > sampleFrame1     = std::make_shared< Frame >( sampleCamera, testImage, maxPyramidLevels, timestampTestImage );
    std::shared_ptr< Feature > sampleFeature1 = std::make_shared< Feature >( sampleFrame1, Eigen::Vector2d( 5.0, 5.0 ), level );

    // Create second frame
    std::shared_ptr< Frame > sampleFrame2     = std::make_shared< Frame >( sampleCamera, testImage, maxPyramidLevels, timestampTestImage );
    std::shared_ptr< Feature > sampleFeature2 = std::make_shared< Feature >( sampleFrame2, Eigen::Vector2d( 20.0, 20.0 ), level );

    Eigen::Vector3d point( 5.0, -2.0, 11.0 );
    std::shared_ptr< Point > samplePoint = std::make_shared< Point >( point );

    // Add two features from two frames
    samplePoint->addFeature( sampleFeature1 );
    samplePoint->addFeature( sampleFeature2 );

    EXPECT_EQ( samplePoint->numberObservation(), 2 );
}

TEST_F( PointTest, TestCloseObservation )
{
    // sample rotation from world to camera coordinate
    Eigen::Matrix3d rotation;
    rotation << -1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0;

    uint8_t level = 0U;
    // Create first frame
    std::shared_ptr< Frame > sampleFrame1     = std::make_shared< Frame >( sampleCamera, testImage, maxPyramidLevels, timestampTestImage );
    sampleFrame1->m_absolutePose              = Sophus::SE3d( rotation, Eigen::Vector3d( 0.0, 0.0, 5.0 ) );
    std::shared_ptr< Feature > sampleFeature1 = std::make_shared< Feature >( sampleFrame1, Eigen::Vector2d( 5.0, 5.0 ), level );

    // Create second frame
    std::shared_ptr< Frame > sampleFrame2     = std::make_shared< Frame >( sampleCamera, testImage, maxPyramidLevels, timestampTestImage );
    sampleFrame2->m_absolutePose              = Sophus::SE3d( rotation, Eigen::Vector3d( 5.0, 0.0, 5.0 ) );
    std::shared_ptr< Feature > sampleFeature2 = std::make_shared< Feature >( sampleFrame2, Eigen::Vector2d( 20.0, 20.0 ), level );

    // Create third frame
    std::shared_ptr< Frame > sampleFrame3 = std::make_shared< Frame >( sampleCamera, testImage, maxPyramidLevels, timestampTestImage );
    sampleFrame3->m_absolutePose          = Sophus::SE3d( rotation, Eigen::Vector3d( 10.0, 0.0, 5.0 ) );

    Eigen::Vector3d point( 0.0, 0.0, 0.0 );
    std::shared_ptr< Point > samplePoint = std::make_shared< Point >( point );

    // Add two features from two frames
    samplePoint->addFeature( sampleFeature1 );
    samplePoint->addFeature( sampleFeature2 );

    std::shared_ptr< Feature > sampleFeature3{ nullptr };
    bool res = samplePoint->getCloseViewObservation( sampleFrame3->cameraInWorld(), sampleFeature3 );
    EXPECT_TRUE( res );
    EXPECT_EQ( sampleFeature3->m_id, sampleFeature2->m_id );
    EXPECT_EQ( sampleFeature3.get(), sampleFeature2.get() );
}
