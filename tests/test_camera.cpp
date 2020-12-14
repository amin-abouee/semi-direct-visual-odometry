#include "pinhole_camera.hpp"

#include <opencv2/highgui.hpp>
#include "gtest/gtest.h"

#include <memory>

static const double testMaxError = 1e-12;

typedef struct PinholeCameraTestParams
{
    size_t width;
    size_t height;
    Eigen::Vector2d focalLength;
    Eigen::Vector2d principalPoint;
} PinholeCameraTestParams;

typedef struct PinholeCameraTestIsInFrameParams
{
    size_t width;
    size_t height;
    Eigen::Vector2d pixelPosition;
    uint8_t level;
    double boundary;
    bool expect;
} PinholeCameraTestIsInFrameParams;

class IsInFrameTestFixture : public ::testing::TestWithParam< PinholeCameraTestIsInFrameParams >
{
protected:
    IsInFrameTestFixture()
    {
    }
    virtual ~IsInFrameTestFixture()
    {
    }

    void SetUp( const PinholeCameraTestIsInFrameParams& par )
    {
        cam = std::make_shared< PinholeCamera >( par.width, par.height, Eigen::Vector2d( 0.0, 0.0 ), Eigen::Vector2d( 0.0, 0.0 ) );
    }

    virtual void TearDown()
    {
    }

    std::shared_ptr< PinholeCamera > cam;
};

void checkCamValues( const PinholeCameraTestParams& params, const PinholeCamera& cam )
{
    EXPECT_EQ( cam.m_width, params.width );
    EXPECT_EQ( cam.m_height, params.height );

    EXPECT_EQ( cam.m_focalLength, params.focalLength );
    EXPECT_EQ( cam.m_principalPoint, params.principalPoint );

    EXPECT_EQ( cam.m_k, ( Eigen::Matrix3d() << params.focalLength.x(), 0, params.principalPoint.x(), 0, params.focalLength.y(),
                          params.principalPoint.y(), 0, 0, 1 )
                          .finished() );

    EXPECT_EQ( cam.m_invK, cam.m_k.inverse() );
}

TEST( PinholeCameraClass, ConstructorAndGetters1 )
{
    PinholeCameraTestParams par = { 10, 20, Eigen::Vector2d( 30.3, 40.4 ), Eigen::Vector2d( 50.5, 60.6 ) };

    PinholeCamera cam( par.width, par.height, par.focalLength, par.principalPoint );

    checkCamValues( par, cam );
}

TEST( PinholeCameraClass, ConstructorAndGetters2 )
{
    PinholeCameraTestParams par = { 10, 20, Eigen::Vector2d( 30.3, 40.4 ), Eigen::Vector2d( 50.5, 60.6 ) };

    PinholeCamera cam( par.width, par.height, par.focalLength.x(), par.focalLength.y(), par.principalPoint.x(), par.principalPoint.y() );

    checkCamValues( par, cam );
}

TEST( PinholeCameraClass, Project2d )
{
    PinholeCameraTestParams par = { 640, 480, Eigen::Vector2d( 30.3, 40.4 ), Eigen::Vector2d( 325.5, 248.8 ) };

    PinholeCamera cam( par.width, par.height, par.focalLength, par.principalPoint );

    const Eigen::Vector3d point3dWCS( 17.7, 28.8, 39.9 );

    // Test projection.
    const Eigen::Vector2d res1 = cam.project2d( point3dWCS );

    EXPECT_DOUBLE_EQ( res1.x(), 338.9413533834586466165 );
    EXPECT_DOUBLE_EQ( res1.y(), 277.9609022556390977443 );

    // Test inversed projection: project2d returns z-normalized result.
    const Eigen::Vector3d res2 = cam.invProject2d( res1 ) * point3dWCS.z();

    EXPECT_NEAR( res2.x(), point3dWCS.x(), testMaxError );
    EXPECT_NEAR( res2.y(), point3dWCS.y(), testMaxError );
    EXPECT_NEAR( res2.z(), point3dWCS.z(), testMaxError );
}

TEST( PinholeCameraClass, Undistortion )
{
    PinholeCameraTestParams par = { 1280, 960, Eigen::Vector2d( 560.33468243, 561.37973145 ),
                                    Eigen::Vector2d( 651.26269237, 499.06652492 ) };

    Eigen::VectorXd dist( 5 );
    dist << -2.32951777e-01, 6.17256346e-02, -1.83274571e-05, 3.39255772e-05, -7.54987702e-03;

    PinholeCamera cam( par.width, par.height, par.focalLength, par.principalPoint, &dist );

    cv::Mat in = cv::imread( TEST_DATA_DIR "camera/undistort_input.png" );

    cv::Mat out;
    cam.undistortImage( in, out );

    cv::Mat ref = cv::imread( TEST_DATA_DIR "camera/undistort_ref.png" );

    cv::Mat diff;
    cv::absdiff( out, ref, diff );

    double min, max;
    cv::minMaxLoc( diff, &min, &max );

    EXPECT_EQ( min, 0 );
    EXPECT_EQ( max, 0 );
}

INSTANTIATE_TEST_CASE_P( PinholeCameraClass,
                         IsInFrameTestFixture,
                         ::testing::Values(
                           // Corners
                           PinholeCameraTestIsInFrameParams{ 4, 3, { 0.0, 0.0 }, 0, 0, true },
                           PinholeCameraTestIsInFrameParams{ 4, 3, { 4.0, 3.0 }, 0, 0, true },
                           PinholeCameraTestIsInFrameParams{ 4, 3, { 4.0, 0.0 }, 0, 0, true },
                           PinholeCameraTestIsInFrameParams{ 4, 3, { 0.0, 3.0 }, 0, 0, true },

                           // Middle
                           PinholeCameraTestIsInFrameParams{ 4, 3, { 1.1, 2.2 }, 0, 0, true },

                           // Just outside the frame
                           PinholeCameraTestIsInFrameParams{ 4, 3, { -0.1, 0.1 }, 0, 0, false },
                           PinholeCameraTestIsInFrameParams{ 4, 3, { 0.1, -0.1 }, 0, 0, false },
                           PinholeCameraTestIsInFrameParams{ 4, 3, { 4.1, 0.1 }, 0, 0, false },
                           PinholeCameraTestIsInFrameParams{ 4, 3, { 0.1, 3.1 }, 0, 0, false },

                           // Boundaries - corners
                           PinholeCameraTestIsInFrameParams{ 4, 3, { 0.5, 0.5 }, 0, 0.5, true },
                           PinholeCameraTestIsInFrameParams{ 4, 3, { 3.5, 0.5 }, 0, 0.5, true },
                           PinholeCameraTestIsInFrameParams{ 4, 3, { 0.5, 2.5 }, 0, 0.5, true },
                           PinholeCameraTestIsInFrameParams{ 4, 3, { 3.5, 2.5 }, 0, 0.5, true },

                           // Boundaries - middle
                           PinholeCameraTestIsInFrameParams{ 4, 3, { 0.6, 0.6 }, 0, 0.5, true },

                           // Boundaries - just outside the frame
                           PinholeCameraTestIsInFrameParams{ 4, 3, { 0.49, 0.49 }, 0, 0.5, false },
                           PinholeCameraTestIsInFrameParams{ 4, 3, { 3.51, 2.00 }, 0, 0.5, false },
                           PinholeCameraTestIsInFrameParams{ 4, 3, { 2.00, 2.51 }, 0, 0.5, false },
                           PinholeCameraTestIsInFrameParams{ 4, 3, { 3.51, 2.51 }, 0, 0.5, false },

                           // Levels - corners
                           PinholeCameraTestIsInFrameParams{ 40, 20, { 0.0, 0.0 }, 1, 0, true },
                           PinholeCameraTestIsInFrameParams{ 40, 20, { 20.0, 10.0 }, 1, 0, true },
                           PinholeCameraTestIsInFrameParams{ 40, 20, { 20.1, 10.0 }, 1, 0, false },
                           PinholeCameraTestIsInFrameParams{ 40, 20, { 20.0, 10.1 }, 1, 0, false },

                           PinholeCameraTestIsInFrameParams{ 40, 20, { 0.0, 0.0 }, 2, 0, true },
                           PinholeCameraTestIsInFrameParams{ 40, 20, { 10.0, 5.0 }, 2, 0, true },
                           PinholeCameraTestIsInFrameParams{ 40, 20, { 10.1, 5.0 }, 2, 0, false },
                           PinholeCameraTestIsInFrameParams{ 40, 20, { 10.0, 5.1 }, 2, 0, false } ) );

TEST_P( IsInFrameTestFixture, Parameterized )
{
    const PinholeCameraTestIsInFrameParams testParam = GetParam();
    SetUp( testParam );

    if ( testParam.level == 0U )
    {
        EXPECT_EQ( cam->isInImageFrame( testParam.pixelPosition, testParam.boundary ), testParam.expect );
    }
    else
    {
        EXPECT_EQ( cam->isInImageFrame( testParam.pixelPosition, testParam.level, testParam.boundary ), testParam.expect );
    }
}
