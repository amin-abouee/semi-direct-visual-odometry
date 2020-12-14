#include <gtest/gtest.h>
#include "image_pyramid.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

using namespace testing;

class ImagePyramidTest : public Test
{
public:
    cv::Mat testImage = cv::Mat( 640, 480, CV_8UC1 );
    cv::Size testImageSize;
    ImagePyramidTest()
    {
        cv::randu( testImage, cv::Scalar( 0 ), cv::Scalar( 255 ) );
        testImageSize = testImage.size();
    }
};

TEST_F( ImagePyramidTest, ConstructorFromLevels )
{
    std::size_t pyramidLevels           = 4U;
    ImagePyramid imagePyramidFromLevels = ImagePyramid( pyramidLevels );
    EXPECT_EQ( pyramidLevels, imagePyramidFromLevels.getAllImages().capacity() );
}

TEST_F( ImagePyramidTest, ConstructorFromImage )
{
    size_t pyramidLevels               = 4U;
    ImagePyramid imagePyramidFromImage = ImagePyramid( testImage, pyramidLevels );
    EXPECT_EQ( pyramidLevels, imagePyramidFromImage.getAllImages().capacity() );
    EXPECT_EQ( pyramidLevels, imagePyramidFromImage.getSizeImagePyramid() );
    EXPECT_EQ( testImageSize, imagePyramidFromImage.getBaseImageSize() );
}

TEST_F( ImagePyramidTest, CreateImagePyramid )
{
    size_t pyramidLevels                = 4U;
    ImagePyramid imagePyramidFromLevels = ImagePyramid( pyramidLevels );
    imagePyramidFromLevels.createImagePyramid( testImage, pyramidLevels );
    EXPECT_EQ( pyramidLevels, imagePyramidFromLevels.getSizeImagePyramid() );
}

TEST_F( ImagePyramidTest, TestGetters )
{
    size_t pyramidLevels               = 4U;
    ImagePyramid imagePyramidFromImage = ImagePyramid( testImage, pyramidLevels );

    cv::Mat baseImage = imagePyramidFromImage.getBaseImage();
    // Get a matrix with differences between input image and base image returned by pyramid
    cv::Mat diff = baseImage != testImage;
    // Check if we have 0 non zero entries, meaning all entries are 0 -> equal
    bool eq = cv::countNonZero( diff ) == 0;
    EXPECT_TRUE( eq );
    EXPECT_EQ( testImageSize, imagePyramidFromImage.getBaseImageSize() );
    for ( int level = 0; level < pyramidLevels; ++level )
    {
        imagePyramidFromImage.getImageAtLevel( level );
        imagePyramidFromImage.getImageSizeAtLevel( level );
    }
}