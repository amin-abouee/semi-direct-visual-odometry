#include "visualization.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "feature.hpp"

void Visualization::visualizeFeaturePoints( const Frame& frame, const std::string& windowsName )
{
    cv::Mat imgBGR;
    cv::cvtColor( frame.m_imagePyramid.getBaseImage(), imgBGR, cv::COLOR_GRAY2BGR );

    const auto szPoints = frame.numberObservation();
    for ( int i( 0 ); i < szPoints; i++ )
    {
        const auto& feature = frame.m_frameFeatures[ i ]->m_feature;
        cv::circle( imgBGR, cv::Point2i( feature.x(), feature.y() ), 2.0, cv::Scalar( 0, 255, 0 ) );
    }
    cv::imshow( windowsName, imgBGR );
}

void Visualization::visualizeFeaturePointsInBothImages( const Frame& refFrame,
                                                        const Frame& curFrame,
                                                        const std::string& windowsName )
{
    cv::Mat refImgBGR;
    cv::cvtColor( refFrame.m_imagePyramid.getBaseImage(), refImgBGR, cv::COLOR_GRAY2BGR );

    auto szPoints = refFrame.numberObservation();
    for ( int i( 0 ); i < szPoints; i++ )
    {
        const auto& feature = refFrame.m_frameFeatures[ i ]->m_feature;
        cv::circle( refImgBGR, cv::Point2i( feature.x(), feature.y() ), 3.0, cv::Scalar( 128, 255, 0 ) );
    }
    // cv::imshow( windowsName, imgBGR );
    cv::Mat curImgBGR;
    cv::cvtColor( curFrame.m_imagePyramid.getBaseImage(), curImgBGR, cv::COLOR_GRAY2BGR );

    szPoints = curFrame.numberObservation();
    for ( int i( 0 ); i < szPoints; i++ )
    {
        const auto& feature = curFrame.m_frameFeatures[ i ]->m_feature;
        cv::circle( curImgBGR, cv::Point2i( feature.x(), feature.y() ), 3.0, cv::Scalar( 0, 128, 255 ) );
    }

    cv::Mat stickImages;
    cv::hconcat( refImgBGR, curImgBGR, stickImages );
    cv::imshow( windowsName, stickImages );
}

void Visualization::visualizeFeaturePointsInBothImagesWithSearchRegion( const Frame& refFrame,
                                                                        const Frame& curFrame,
                                                                        const uint16_t& patchSize,
                                                                        const std::string& windowsName )
{
    const uint16_t halfPatch = patchSize / 2;
    cv::Mat refImgBGR;
    cv::cvtColor( refFrame.m_imagePyramid.getBaseImage(), refImgBGR, cv::COLOR_GRAY2BGR );

    auto szPoints = refFrame.numberObservation();
    for ( int i( 0 ); i < szPoints; i++ )
    {
        const auto& feature = refFrame.m_frameFeatures[ i ]->m_feature;
        cv::circle( refImgBGR, cv::Point2i( feature.x(), feature.y() ), 3.0, cv::Scalar( 128, 255, 0 ) );
    }
    // cv::imshow( windowsName, imgBGR );
    cv::Mat curImgBGR;
    cv::cvtColor( curFrame.m_imagePyramid.getBaseImage(), curImgBGR, cv::COLOR_GRAY2BGR );

    szPoints = curFrame.numberObservation();
    for ( int i( 0 ); i < szPoints; i++ )
    {
        const auto& feature = curFrame.m_frameFeatures[ i ]->m_feature;
        cv::circle( curImgBGR, cv::Point2i( feature.x(), feature.y() ), 3.0, cv::Scalar( 0, 128, 255 ) );
        cv::rectangle( curImgBGR, cv::Point2i( feature.x() - halfPatch, feature.y() - halfPatch ),
                       cv::Point2i( feature.x() + halfPatch, feature.y() + halfPatch ), cv::Scalar( 0, 128, 255 ) );
    }

    cv::Mat stickImages;
    cv::hconcat( refImgBGR, curImgBGR, stickImages );
    cv::imshow( windowsName, stickImages );
}

void Visualization::visualizeFeaturePoints( const cv::Mat& img, const Frame& frame, const std::string& windowsName )
{
    cv::Mat normMag, imgBGR;
    cv::normalize( img, normMag, 0, 255, cv::NORM_MINMAX, CV_8UC1 );
    cv::cvtColor( normMag, imgBGR, cv::COLOR_GRAY2BGR );

    const auto szPoints = frame.m_frameFeatures.size();
    for ( int i( 0 ); i < szPoints; i++ )
    {
        const auto& feature = frame.m_frameFeatures[ i ]->m_feature;
        cv::circle( imgBGR, cv::Point2i( feature.x(), feature.y() ), 2.0, cv::Scalar( 0, 255, 0 ) );
    }
    cv::imshow( windowsName, imgBGR );
}

void Visualization::visualizeGrayImage( const cv::Mat& img, const std::string& windowsName )
{
    // double min, max;
    // cv::minMaxLoc( img, &min, &max );
    // std::cout << "min: " << min << ", max: " << max << std::endl;

    // cv::Mat grad;
    // cv::addWeighted( absDx, 0.5, absDy, 0.5, 0, grad );
    // cv::imshow("grad_mag_weight", grad);

    // cv::Mat absoluteGrad = absDx + absDy;
    // cv::imshow("grad_mag_abs", absoluteGrad);

    // cv::Mat absMag;
    // cv::convertScaleAbs(mag, absMag);
    // cv::imshow("grad_mag_scale", absMag);

    cv::Mat normMag;
    cv::normalize( img, normMag, 0, 255, cv::NORM_MINMAX, CV_8UC1 );
    cv::imshow( windowsName, normMag );
}

void Visualization::visualizeHSVColoredImage( const cv::Mat& img, const std::string& windowsName )
{
    // https://realpython.com/python-opencv-color-spaces/
    // https://stackoverflow.com/questions/23001512/c-and-opencv-get-and-set-pixel-color-to-mat
    // https://answers.opencv.org/question/178766/adjust-hue-and-saturation-like-photoshop/
    // http://colorizer.org/
    // https://toolstud.io/color/rgb.php?rgb_r=0&rgb_g=255&rgb_b=0&convert=rgbdec
    // https://answers.opencv.org/question/191488/create-a-hsv-range-palette/

    // https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html

    cv::Mat normMag, imgBGR, imgHSV;
    cv::normalize( img, normMag, 0, 255, cv::NORM_MINMAX, CV_8UC1 );
    cv::cvtColor( normMag, imgBGR, cv::COLOR_GRAY2BGR );
    cv::cvtColor( imgBGR, imgHSV, cv::COLOR_BGR2HSV );

    const float minMagnitude = 75.0;
    for ( int i( 0 ); i < imgHSV.rows; i++ )
    {
        for ( int j( 0 ); j < imgHSV.cols; j++ )
        {
            if ( img.at< float >( i, j ) >= minMagnitude )
            {
                cv::Vec3b& px    = imgHSV.at< cv::Vec3b >( cv::Point( j, i ) );
                cv::Scalar color = generateColor( minMagnitude, 255.0, img.at< float >( i, j ) - minMagnitude );
                px[ 0 ]          = color[ 0 ];
                px[ 1 ]          = color[ 1 ];
                px[ 2 ]          = color[ 2 ];
            }
        }
    }
    cv::Mat imgHSVNew;
    cv::cvtColor( imgHSV, imgHSVNew, cv::COLOR_HSV2BGR );
    cv::imshow( windowsName, imgHSVNew );
}

void Visualization::visualizeEpipole( const Frame& frame, const Eigen::Vector3d& vec, const std::string& windowsName )
{
    // https://answers.opencv.org/question/182587/how-to-draw-epipolar-line/
    cv::Mat imgBGR = getBGRImage( frame.m_imagePyramid.getBaseImage() );

    const Eigen::Vector2d projected = frame.m_camera->project2d( vec );
    cv::circle( imgBGR, cv::Point2i( projected.x(), projected.y() ), 5.0, cv::Scalar( 0, 255, 165 ) );
    cv::imshow( windowsName, imgBGR );
}

void Visualization::visualizeEpipolarLine( const Frame& frame,
                                           const Eigen::Vector3d& vec,
                                           const Eigen::Matrix3d& F,
                                           const std::string& windowsName )
{
    cv::Mat imgBGR = getBGRImage( frame.m_imagePyramid.getBaseImage() );

    Eigen::Vector3d line = F * vec;

    double nu = line( 0 ) * line( 0 ) + line( 1 ) * line( 1 );
    nu        = 1 / std::sqrt( nu );
    line *= nu;

    const cv::Point p1( 0, -line( 2 ) / line( 1 ) );
    const cv::Point p2( imgBGR.cols - 1, -( line( 2 ) + line( 0 ) * ( imgBGR.cols - 1 ) ) / line( 1 ) );
    cv::line( imgBGR, p1, p2, cv::Scalar( 0, 160, 200 ) );
    cv::imshow( windowsName, imgBGR );
}

void Visualization::visualizeEpipolarLine( const Frame& curFrame,
                                           const Eigen::Vector3d& normalizedVec,
                                           const double minDepth,
                                           const double maxDepth,
                                           const std::string& windowsName )
{
    cv::Mat imgBGR               = getBGRImage( curFrame.m_imagePyramid.getBaseImage() );
    const Sophus::SE3d T_Pre2Cur = Sophus::SE3d().inverse() * curFrame.m_TransW2F;
    // std::cout << "T_Pre2Cur: " << T_Pre2Cur.params().transpose() << std::endl;
    const Eigen::Vector2d point1 = curFrame.camera2image( T_Pre2Cur * ( normalizedVec * minDepth ) );
    const Eigen::Vector2d point2 = curFrame.camera2image( T_Pre2Cur * ( normalizedVec * maxDepth ) );
    // std::cout << "Position point 1: " << point1.transpose() << std::endl;
    // std::cout << "Position point 2: " << point2.transpose() << std::endl;
    cv::line( imgBGR, cv::Point( point1.x(), point1.y() ), cv::Point( point2.x(), point2.y() ),
              cv::Scalar( 200, 160, 10 ) );
    cv::imshow( windowsName, imgBGR );
}

void Visualization::visualizeEpipolarLine( const Frame& refFrame,
                                           const Frame& curFrame,
                                           const Eigen::Vector2d& feature,
                                           const double minDepth,
                                           const double maxDepth,
                                           const std::string& windowsName )
{
    cv::Mat refImgBGR            = getBGRImage( refFrame.m_imagePyramid.getBaseImage() );
    cv::Mat curImgBGR            = getBGRImage( curFrame.m_imagePyramid.getBaseImage() );
    const Sophus::SE3d T_Ref2Cur = refFrame.m_TransW2F.inverse() * curFrame.m_TransW2F;
    // std::cout << "Visualization 1 -> 2: " << T_Ref2Cur.params().transpose() << std::endl;
    // std::cout << "refFrame.image2camera(feature): " << refFrame.image2camera(feature, 1.0).transpose() << std::endl;
    const Eigen::Vector2d point1 = curFrame.camera2image( T_Ref2Cur * ( refFrame.image2camera( feature, minDepth ) ) );
    const Eigen::Vector2d point2 = curFrame.camera2image( T_Ref2Cur * ( refFrame.image2camera( feature, maxDepth ) ) );
    const Eigen::Vector2d C      = curFrame.camera2image( T_Ref2Cur * ( refFrame.image2camera( feature, 0.0 ) ) );
    // std::cout << "Position point 1: " << point1.transpose() << std::endl;
    // std::cout << "Position point 2: " << point2.transpose() << std::endl;
    cv::circle( refImgBGR, cv::Point2i( feature.x(), feature.y() ), 5.0, cv::Scalar( 255, 10, 255 ) );
    cv::line( curImgBGR, cv::Point( point1.x(), point1.y() ), cv::Point( point2.x(), point2.y() ),
              cv::Scalar( 200, 160, 10 ) );
    cv::circle( curImgBGR, cv::Point2i( C.x(), C.y() ), 5.0, cv::Scalar( 0, 255, 165 ) );

    cv::Mat stickImages;
    cv::hconcat( refImgBGR, curImgBGR, stickImages );
    cv::imshow( windowsName, stickImages );
}

void Visualization::visualizeEpipolarLinesWithFundamenalMatrix( const Frame& frame,
                                                                const cv::Mat& currentImg,
                                                                const Eigen::Matrix3d& F,
                                                                const std::string& windowsName )
{
    cv::Mat imgBGR      = getBGRImage( currentImg );
    const auto szPoints = frame.numberObservation();

    for ( int i( 0 ); i < szPoints; i++ )
    {
        const auto& feature  = frame.m_frameFeatures[ i ]->m_feature;
        Eigen::Vector3d line = F * Eigen::Vector3d( feature.x(), feature.y(), 1.0 );
        double nu            = line( 0 ) * line( 0 ) + line( 1 ) * line( 1 );
        nu                   = 1 / std::sqrt( nu );
        line *= nu;
        const cv::Point p1( 0, -line( 2 ) / line( 1 ) );
        const cv::Point p2( imgBGR.cols - 1, -( line( 2 ) + line( 0 ) * ( imgBGR.cols - 1 ) ) / line( 1 ) );
        cv::line( imgBGR, p1, p2, cv::Scalar( 0, 160, 200 ) );
    }
    cv::circle( imgBGR, cv::Point2i( 1004, 119 ), 5.0, cv::Scalar( 0, 255, 165 ) );
    cv::imshow( windowsName, imgBGR );
}

void Visualization::visualizeEpipolarLinesWithEssentialMatrix( const Frame& frame,
                                                               const cv::Mat& currentImg,
                                                               const Eigen::Matrix3d& E,
                                                               const std::string& windowsName )
{
    const Eigen::Matrix3d F = frame.m_camera->invK().transpose() * E * frame.m_camera->invK();
    visualizeEpipolarLinesWithFundamenalMatrix( frame, currentImg, F, windowsName );
}

cv::Scalar Visualization::generateColor( const double min, const double max, const float value )
{
    int hue = ( 120 / ( max - min ) ) * value;
    return cv::Scalar( hue, 100, 100 );
}

cv::Mat Visualization::getBGRImage( const cv::Mat& img )
{
    cv::Mat imgBGR;
    if ( img.channels() == 1 )
        cv::cvtColor( img, imgBGR, cv::COLOR_GRAY2BGR );
    else
        imgBGR = img.clone();
    return imgBGR;
}