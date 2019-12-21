#include "visualization.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "feature.hpp"
#include "algorithm.hpp"
#include "utils.hpp"

// std::map< std::string, cv::Scalar > colors{
//   {"red", cv::Scalar( 65, 82, 226 )},     {"pink", cv::Scalar( 101, 57, 215 )},
//   {"purple", cv::Scalar( 170, 55, 144 )}, {"deep-purple", cv::Scalar( 177, 65, 96 )},
//   {"indigo", cv::Scalar( 175, 84, 65 )},  {"blue", cv::Scalar( 236, 150, 70 )},
//   {"cyan", cv::Scalar( 209, 186, 83 )},   {"aqua", cv::Scalar( 253, 252, 115 )},
//   {"teal", cv::Scalar( 136, 148, 65 )},   {"green", cv::Scalar( 92, 172, 103 )},
//   {"lime", cv::Scalar( 89, 218, 209 )},   {"yellow", cv::Scalar( 96, 234, 253 )},
//   {"amber", cv::Scalar( 68, 194, 246 )},  {"orange", cv::Scalar( 56, 156, 242 )},
//   {"brown", cv::Scalar( 74, 86, 116 )},   {"gray", cv::Scalar( 158, 158, 158 )},
//   {"black", cv::Scalar( 0, 0, 0 )},       {"deep-orange", cv::Scalar( 55, 99, 237 )},
//   {"white", cv::Scalar( 356, 256, 256 )}};

void visualization::featurePoints( const Frame& frame, const std::string& windowsName )
{
    cv::Mat imgBGR;
    cv::cvtColor( frame.m_imagePyramid.getBaseImage(), imgBGR, cv::COLOR_GRAY2BGR );

    const auto szPoints = frame.numberObservation();
    for ( std::size_t i( 0 ); i < szPoints; i++ )
    {
        const auto& feature = frame.m_frameFeatures[ i ]->m_feature;
        cv::circle( imgBGR, cv::Point2d( feature.x(), feature.y() ), 2.0, cv::Scalar( 0, 255, 0 ) );
    }
    cv::imshow( windowsName, imgBGR );
}

void visualization::featurePointsInBothImages( const Frame& refFrame,
                                               const Frame& curFrame,
                                               const std::string& windowsName )
{
    // cv::Mat refImgBGR;
    // cv::cvtColor( refFrame.m_imagePyramid.getBaseImage(), refImgBGR, cv::COLOR_GRAY2BGR );

    // auto szPoints = refFrame.numberObservation();
    // for ( std::size_t i( 0 ); i < szPoints; i++ )
    // {
    //     const auto& feature = refFrame.m_frameFeatures[ i ]->m_feature;
    //     cv::circle( refImgBGR, cv::Point2i( feature.x(), feature.y() ), 3.0, cv::Scalar( 255, 0, 255 ) );
    // }
    // // cv::imshow( windowsName, imgBGR );
    // cv::Mat curImgBGR;
    // cv::cvtColor( curFrame.m_imagePyramid.getBaseImage(), curImgBGR, cv::COLOR_GRAY2BGR );

    // szPoints = curFrame.numberObservation();
    // for ( std::size_t i( 0 ); i < szPoints; i++ )
    // {
    //     const auto& feature = curFrame.m_frameFeatures[ i ]->m_feature;
    //     cv::circle( curImgBGR, cv::Point2i( feature.x(), feature.y() ), 3.0, cv::Scalar( 0, 128, 255 ) );
    // }

    // cv::Mat stickImages;
    // cv::hconcat( refImgBGR, curImgBGR, stickImages );
    // cv::imshow( windowsName, stickImages );
    visualization::featurePointsInBothImagesWithSearchRegion( refFrame, curFrame, 0, windowsName );
}

void visualization::featurePointsInBothImagesWithSearchRegion( const Frame& refFrame,
                                                               const Frame& curFrame,
                                                               const uint16_t& patchSize,
                                                               const std::string& windowsName )
{
    const uint16_t halfPatch = patchSize / 2;
    cv::Mat refImgBGR;
    cv::cvtColor( refFrame.m_imagePyramid.getBaseImage(), refImgBGR, cv::COLOR_GRAY2BGR );

    auto szPoints = refFrame.numberObservation();
    for ( std::size_t i( 0 ); i < szPoints; i++ )
    {
        const auto& feature = refFrame.m_frameFeatures[ i ]->m_feature;
        // cv::circle( refImgBGR, cv::Point2i( feature.x(), feature.y() ), 3.0, cv::Scalar( 255, 0, 255 ) );
        cv::circle( refImgBGR, cv::Point2d( feature.x(), feature.y() ), 3.0, colors.at( "pink" ) );
    }

    cv::Mat curImgBGR;
    cv::cvtColor( curFrame.m_imagePyramid.getBaseImage(), curImgBGR, cv::COLOR_GRAY2BGR );
    szPoints = curFrame.numberObservation();
    for ( std::size_t i( 0 ); i < szPoints; i++ )
    {
        const auto& feature = curFrame.m_frameFeatures[ i ]->m_feature;
        // cv::circle( curImgBGR, cv::Point2i( feature.x(), feature.y() ), 3.0, cv::Scalar( 0, 128, 255 ) );
        cv::circle( curImgBGR, cv::Point2d( feature.x(), feature.y() ), 3.0, colors.at( "orange" ) );
        if ( patchSize > 0 )
        {
            cv::rectangle( curImgBGR, cv::Point2d( feature.x() - halfPatch, feature.y() - halfPatch ),
                           cv::Point2d( feature.x() + halfPatch, feature.y() + halfPatch ), colors.at( "orange" ) );
        }
    }

    cv::Mat stickImages;
    cv::hconcat( refImgBGR, curImgBGR, stickImages );
    cv::imshow( windowsName, stickImages );
}

void visualization::featurePoints( const cv::Mat& img, const Frame& frame, const std::string& windowsName )
{
    cv::Mat normMag, imgBGR;
    cv::normalize( img, normMag, 0, 255, cv::NORM_MINMAX, CV_8UC1 );
    cv::cvtColor( normMag, imgBGR, cv::COLOR_GRAY2BGR );

    const auto szPoints = frame.m_frameFeatures.size();
    for ( std::size_t i( 0 ); i < szPoints; i++ )
    {
        const auto& feature = frame.m_frameFeatures[ i ]->m_feature;
        cv::circle( imgBGR, cv::Point2d( feature.x(), feature.y() ), 2.0, colors.at( "teal" ) );
    }
    cv::imshow( windowsName, imgBGR );
}

void visualization::featurePointsInGrid( const cv::Mat& img, const Frame& frame, const int32_t gridSize, const std::string& windowsName )
{
    cv::Mat normMag, imgBGR;
    cv::normalize( img, normMag, 0, 255, cv::NORM_MINMAX, CV_8UC1 );
    cv::cvtColor( normMag, imgBGR, cv::COLOR_GRAY2BGR );

    const auto szPoints = frame.m_frameFeatures.size();
    for ( std::size_t i( 0 ); i < szPoints; i++ )
    {
        const auto& feature = frame.m_frameFeatures[ i ]->m_feature;
        cv::circle( imgBGR, cv::Point2d( feature.x(), feature.y() ), 2.0, colors.at( "amber" ) );
    }

    const int width  = img.cols;
    const int height = img.rows;

    const int cols = width / gridSize;
    const int rows = height / gridSize;
    for (int r(1); r <= rows; r++)
    {
        cv::line( imgBGR, cv::Point2i(0, r * gridSize), cv::Point2i(width, r * gridSize), colors.at( "amber" ) );
    }
    
    for (int c(1); c <= cols; c++)
    {
        cv::line( imgBGR, cv::Point2i(c * gridSize, 0), cv::Point2i(c * gridSize, height), colors.at( "amber" ) );
    }

    cv::imshow( windowsName, imgBGR );
}

void visualization::grayImage( const cv::Mat& img, const std::string& windowsName )
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

void visualization::HSVColoredImage( const cv::Mat& img, const std::string& windowsName )
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
                cv::Scalar color = generateColor( minMagnitude, 255.0f, img.at< float >( i, j ) - minMagnitude );
                px[ 0 ]          = static_cast<uchar>(color[ 0 ]);
                px[ 1 ]          = static_cast<uchar>(color[ 1 ]);
                px[ 2 ]          = static_cast<uchar>(color[ 2 ]);
            }
        }
    }
    cv::Mat imgHSVNew;
    cv::cvtColor( imgHSV, imgHSVNew, cv::COLOR_HSV2BGR );
    cv::imshow( windowsName, imgHSVNew );
}

void visualization::epipole( const Frame& frame, const Eigen::Vector3d& vec, const std::string& windowsName )
{
    // https://answers.opencv.org/question/182587/how-to-draw-epipolar-line/
    cv::Mat imgBGR = getBGRImage( frame.m_imagePyramid.getBaseImage() );

    const Eigen::Vector2d projected = frame.m_camera->project2d( vec );
    cv::circle( imgBGR, cv::Point2d( projected.x(), projected.y() ), 8.0, colors.at( "lime" ) );
    cv::imshow( windowsName, imgBGR );
}

void visualization::epipolarLine( const Frame& frame,
                                  const Eigen::Vector3d& vec,
                                  const Eigen::Matrix3d& F,
                                  const std::string& windowsName )
{
    cv::Mat imgBGR = getBGRImage( frame.m_imagePyramid.getBaseImage() );

    Eigen::Vector3d line = F * vec;

    double nu = line( 0 ) * line( 0 ) + line( 1 ) * line( 1 );
    nu        = 1 / std::sqrt( nu );
    line *= nu;

    const cv::Point2d p1( 0.0, -line( 2 ) / line( 1 ) );
    const cv::Point2d p2( imgBGR.cols - 1, -( line( 2 ) + line( 0 ) * ( imgBGR.cols - 1 ) ) / line( 1 ) );
    cv::line( imgBGR, p1, p2, colors.at( "amber" ) );
    cv::imshow( windowsName, imgBGR );
}

void visualization::epipolarLine( const Frame& curFrame,
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
    cv::line( imgBGR, cv::Point2d( point1.x(), point1.y() ), cv::Point2d( point2.x(), point2.y() ), colors.at( "amber" ) );
    //   cv::Scalar( 200, 160, 10 ) );
    cv::imshow( windowsName, imgBGR );
}

void visualization::epipolarLineBothImages( const Frame& refFrame,
                                            const Frame& curFrame,
                                            const Eigen::Vector2d& feature,
                                            const double minDepth,
                                            const double maxDepth,
                                            const std::string& windowsName )
{
    cv::Mat refImgBGR            = getBGRImage( refFrame.m_imagePyramid.getBaseImage() );
    cv::Mat curImgBGR            = getBGRImage( curFrame.m_imagePyramid.getBaseImage() );
    const Sophus::SE3d T_Ref2Cur = refFrame.m_TransW2F.inverse() * curFrame.m_TransW2F;
    // std::cout << "visualization 1 -> 2: " << T_Ref2Cur.params().transpose() << std::endl;
    // std::cout << "refFrame.image2camera(feature): " << refFrame.image2camera(feature, 1.0).transpose() << std::endl;
    const Eigen::Vector2d point1 = curFrame.camera2image( T_Ref2Cur * ( refFrame.image2camera( feature, minDepth ) ) );
    const Eigen::Vector2d point2 = curFrame.camera2image( T_Ref2Cur * ( refFrame.image2camera( feature, maxDepth ) ) );
    const Eigen::Vector2d C      = curFrame.camera2image( T_Ref2Cur * ( refFrame.image2camera( feature, 0.0 ) ) );
    // std::cout << "Position point 1: " << point1.transpose() << std::endl;
    // std::cout << "Position point 2: " << point2.transpose() << std::endl;
    // cv::circle( refImgBGR, cv::Point2i( feature.x(), feature.y() ), 5.0, cv::Scalar( 255, 10, 255 ) );
    cv::circle( refImgBGR, cv::Point2d( feature.x(), feature.y() ), 5.0, colors.at( "purple" ) );
    cv::line( curImgBGR, cv::Point2d( point1.x(), point1.y() ), cv::Point2d( point2.x(), point2.y() ),
              colors.at( "amber" ) );
    //   cv::Scalar( 200, 160, 10 ) );
    cv::circle( curImgBGR, cv::Point2d( C.x(), C.y() ), 5.0, colors.at( "orange" ) );
    // cv::circle( curImgBGR, cv::Point2i( C.x(), C.y() ), 5.0, cv::Scalar( 0, 255, 165 ) );

    cv::Mat stickImages;
    cv::hconcat( refImgBGR, curImgBGR, stickImages );
    cv::imshow( windowsName, stickImages );
}

void visualization::epipolarLinesWithDepth( const Frame& refFrame,
                                            const Frame& curFrame,
                                            const Eigen::VectorXd& depths,
                                            const double sigma,
                                            const std::string& windowsName )
{
    cv::Mat refImgBGR;
    cv::cvtColor( refFrame.m_imagePyramid.getBaseImage(), refImgBGR, cv::COLOR_GRAY2BGR );
    cv::Mat curImgBGR;
    cv::cvtColor( curFrame.m_imagePyramid.getBaseImage(), curImgBGR, cv::COLOR_GRAY2BGR );
    const Sophus::SE3d T_Pre2Cur = refFrame.m_TransW2F.inverse() * curFrame.m_TransW2F;
    const Eigen::Matrix3d E = T_Pre2Cur.rotationMatrix() * algorithm::hat(T_Pre2Cur.translation());
    std::cout << "E matrix: " << E.format( utils::eigenFormat() ) << std::endl;
    const Eigen::Matrix3d F = refFrame.m_camera->invK().transpose() * E * curFrame.m_camera->invK();
    std::cout << "F matrix: " << F.format( utils::eigenFormat() ) << std::endl;
    double minDepth              = 0.0;
    double maxDepth              = 0.0;
    const Eigen::Vector2d C = curFrame.camera2image( curFrame.cameraInWorld() );

    auto szPoints = refFrame.numberObservation();
    for ( std::size_t i( 0 ); i < szPoints; i++ )
    {
        const auto& reFeature = refFrame.m_frameFeatures[ i ]->m_feature;
        cv::circle( refImgBGR, cv::Point2d( reFeature.x(), reFeature.y() ), 5.0, colors.at( "pink" ) );

        Eigen::Vector3d line = F * Eigen::Vector3d( reFeature.x(), reFeature.y(), 1.0 );
        double nu            = line( 0 ) * line( 0 ) + line( 1 ) * line( 1 );
        nu                   = 1 / std::sqrt( nu );
        line *= nu;

        const auto& normalizedVec = refFrame.m_frameFeatures[ i ]->m_bearingVec;
        minDepth                  = depths( i ) - sigma;
        maxDepth                  = depths( i ) + sigma;
        // std::cout << "T_Pre2Cur: " << T_Pre2Cur.params().transpose() << std::endl;
        Eigen::Vector2d pointMin = curFrame.camera2image( T_Pre2Cur * ( normalizedVec * minDepth ) );
        Eigen::Vector2d pointMax = curFrame.camera2image( T_Pre2Cur * ( normalizedVec * maxDepth ) );
        // std::cout << "Position point 1: " << pointMin.transpose() << std::endl;
        // std::cout << "Position point 2: " << pointMax.transpose() << std::endl;
        // cv::line( curImgBGR, cv::Point2d( pointMin.x(), pointMin.y() ), cv::Point2d( pointMax.x(), pointMax.y() ),
        //   colors.at( "amber" ) );
        const Eigen::Vector2d pointCenter = curFrame.camera2image( T_Pre2Cur * ( normalizedVec * depths( i ) ) );
        cv::circle( curImgBGR, cv::Point2d( pointCenter.x(), pointCenter.y() ), 5.0, colors.at( "orange" ) );

        pointMin.x() = pointCenter.x() - sigma;
        pointMin.y() = (line(0) * pointMin.x() + line(2))/(-line(1));

        pointMax.x() = pointCenter.x() + sigma;
        pointMax.y() = (line(0) * pointMax.x() + line(2))/(-line(1));
        cv::line( curImgBGR, cv::Point2d( pointMin.x(), pointMin.y() ), cv::Point2d( pointMax.x(), pointMax.y() ),
          colors.at( "amber" ) );

        const auto& curFeature = curFrame.m_frameFeatures[ i ]->m_feature;
        cv::circle( curImgBGR, cv::Point2d( curFeature.x(), curFeature.y() ), 8.0, colors.at( "blue" ) );
        // std::cout << "idx: " << i << ", Pt ref: " << refFrame.m_frameFeatures[ i ]->m_feature.transpose()
                //   << ", error: " << ( pointCenter - feature ).norm() << ", depth: " << depths( i ) << std::endl;
        // const Eigen::Vector2d C      = curFrame.world2image(curFrame.cameraInWorld());
    }
    cv::circle( curImgBGR, cv::Point2d( C.x(), C.y() ), 8.0, colors.at( "red" ) );

    cv::Mat stickImages;
    cv::hconcat( refImgBGR, curImgBGR, stickImages );
    cv::imshow( windowsName, stickImages );
}

void visualization::epipolarLinesWithFundamentalMatrix( const Frame& frame,
                                                        const cv::Mat& currentImg,
                                                        const Eigen::Matrix3d& F,
                                                        const std::string& windowsName )
{
    cv::Mat imgBGR      = getBGRImage( currentImg );
    const auto szPoints = frame.numberObservation();

    for ( std::size_t i( 0 ); i < szPoints; i++ )
    {
        const auto& feature  = frame.m_frameFeatures[ i ]->m_feature;
        Eigen::Vector3d line = F * Eigen::Vector3d( feature.x(), feature.y(), 1.0 );
        double nu            = line( 0 ) * line( 0 ) + line( 1 ) * line( 1 );
        nu                   = 1 / std::sqrt( nu );
        line *= nu;
        const cv::Point2d p1( 0.0, -line( 2 ) / line( 1 ) );
        const cv::Point2d p2( imgBGR.cols - 1, -( line( 2 ) + line( 0 ) * ( imgBGR.cols - 1 ) ) / line( 1 ) );
        // cv::line( imgBGR, p1, p2, cv::Scalar( 0, 160, 200 ) );
        cv::line( imgBGR, p1, p2, colors.at( "deep-orange" ) );
    }
    // cv::circle( imgBGR, cv::Point2i( 1004, 119 ), 5.0, cv::Scalar( 0, 255, 165 ) );
    cv::imshow( windowsName, imgBGR );
}

void visualization::epipolarLinesWithPointsWithFundamentalMatrix( const Frame& refFrame,
                                                                  const Frame& curFrame,
                                                                  const Eigen::Matrix3d& F,
                                                                  const std::string& windowsName )
{
    cv::Mat imgBGR = getBGRImage( curFrame.m_imagePyramid.getBaseImage() );
    // const Sophus::SE3d T_Ref2Cur = refFrame.m_TransW2F.inverse() * curFrame.m_TransW2F;
    const Eigen::Vector2d C = curFrame.camera2image( curFrame.cameraInWorld() );
    const auto szPoints     = refFrame.numberObservation();

    for ( std::size_t i( 0 ); i < szPoints; i++ )
    {
        const auto& refFeature = refFrame.m_frameFeatures[ i ]->m_feature;
        const auto& curFeature = curFrame.m_frameFeatures[ i ]->m_feature;
        Eigen::Vector3d line   = F * Eigen::Vector3d( refFeature.x(), refFeature.y(), 1.0 );
        double nu              = line( 0 ) * line( 0 ) + line( 1 ) * line( 1 );
        nu                     = 1 / std::sqrt( nu );
        line *= nu;
        const cv::Point2d p1( 0.0, -line( 2 ) / line( 1 ) );
        const cv::Point2d p2( imgBGR.cols - 1, -( line( 2 ) + line( 0 ) * ( imgBGR.cols - 1 ) ) / line( 1 ) );
        cv::circle( imgBGR, cv::Point2d( curFeature.x(), curFeature.y() ), 5.0, colors.at( "lime" ) );
        cv::line( imgBGR, p1, p2, colors.at( "teal" ) );
    }
    cv::circle( imgBGR, cv::Point2d( C.x(), C.y() ), 8.0, colors.at( "red" ) );
    cv::imshow( windowsName, imgBGR );
}

void visualization::epipolarLinesWithEssentialMatrix( const Frame& frame,
                                                      const cv::Mat& currentImg,
                                                      const Eigen::Matrix3d& E,
                                                      const std::string& windowsName )
{
    const Eigen::Matrix3d F = frame.m_camera->invK().transpose() * E * frame.m_camera->invK();
    visualization::epipolarLinesWithFundamentalMatrix( frame, currentImg, F, windowsName );
}

cv::Scalar visualization::generateColor( const float min, const float max, const float value )
{
    uint8_t hue = static_cast<uint8_t>(( 120.f / ( max - min ) ) * value);
    return cv::Scalar( hue, 100, 100 );
}

cv::Mat visualization::getBGRImage( const cv::Mat& img )
{
    cv::Mat imgBGR;
    if ( img.channels() == 1 )
        cv::cvtColor( img, imgBGR, cv::COLOR_GRAY2BGR );
    else
        imgBGR = img.clone();
    return imgBGR;
}