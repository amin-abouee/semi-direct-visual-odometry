#include "visualization.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "algorithm.hpp"
#include "feature.hpp"
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

void visualization::featurePointsInBothImages( const Frame& refFrame, const Frame& curFrame, const std::string& windowsName )
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
    for ( int r( 1 ); r <= rows; r++ )
    {
        cv::line( imgBGR, cv::Point2i( 0, r * gridSize ), cv::Point2i( width, r * gridSize ), colors.at( "amber" ) );
    }

    for ( int c( 1 ); c <= cols; c++ )
    {
        cv::line( imgBGR, cv::Point2i( c * gridSize, 0 ), cv::Point2i( c * gridSize, height ), colors.at( "amber" ) );
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
                px[ 0 ]          = static_cast< uchar >( color[ 0 ] );
                px[ 1 ]          = static_cast< uchar >( color[ 1 ] );
                px[ 2 ]          = static_cast< uchar >( color[ 2 ] );
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

void visualization::epipolarLine( const Frame& frame, const Eigen::Vector3d& vec, const Eigen::Matrix3d& F, const std::string& windowsName )
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
    cv::line( curImgBGR, cv::Point2d( point1.x(), point1.y() ), cv::Point2d( point2.x(), point2.y() ), colors.at( "amber" ) );
    //   cv::Scalar( 200, 160, 10 ) );
    cv::circle( curImgBGR, cv::Point2d( C.x(), C.y() ), 5.0, colors.at( "orange" ) );
    // cv::circle( curImgBGR, cv::Point2i( C.x(), C.y() ), 5.0, cv::Scalar( 0, 255, 165 ) );

    cv::Mat stickImages;
    cv::hconcat( refImgBGR, curImgBGR, stickImages );
    cv::imshow( windowsName, stickImages );
}

void visualization::epipolarLinesWithDepth(
  const Frame& refFrame, const Frame& curFrame, const Eigen::VectorXd& depths, const double sigma, const std::string& windowsName )
{
    cv::Mat refImgBGR;
    cv::cvtColor( refFrame.m_imagePyramid.getBaseImage(), refImgBGR, cv::COLOR_GRAY2BGR );
    cv::Mat curImgBGR;
    cv::cvtColor( curFrame.m_imagePyramid.getBaseImage(), curImgBGR, cv::COLOR_GRAY2BGR );
    const Sophus::SE3d T_Pre2Cur = refFrame.m_TransW2F.inverse() * curFrame.m_TransW2F;
    const Eigen::Matrix3d E      = T_Pre2Cur.rotationMatrix() * algorithm::hat( T_Pre2Cur.translation() );
    std::cout << "E matrix: " << E.format( utils::eigenFormat() ) << std::endl;
    const Eigen::Matrix3d F = refFrame.m_camera->invK().transpose() * E * curFrame.m_camera->invK();
    std::cout << "F matrix: " << F.format( utils::eigenFormat() ) << std::endl;
    double minDepth         = 0.0;
    double maxDepth         = 0.0;
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
        Eigen::Vector2d pointMin = curFrame.camera2image( normalizedVec * minDepth );
        Eigen::Vector2d pointMax = curFrame.camera2image( normalizedVec * maxDepth );
        // std::cout << "Position point 1: " << pointMin.transpose() << std::endl;
        // std::cout << "Position point 2: " << pointMax.transpose() << std::endl;
        // cv::line( curImgBGR, cv::Point2d( pointMin.x(), pointMin.y() ), cv::Point2d( pointMax.x(), pointMax.y() ),
        //   colors.at( "amber" ) );
        const Eigen::Vector2d pointCenter = curFrame.camera2image( normalizedVec * depths( i ) );
        cv::circle( curImgBGR, cv::Point2d( pointCenter.x(), pointCenter.y() ), 5.0, colors.at( "orange" ) );

        pointMin.x() = pointCenter.x() - sigma;
        pointMin.y() = ( line( 0 ) * pointMin.x() + line( 2 ) ) / ( -line( 1 ) );

        pointMax.x() = pointCenter.x() + sigma;
        pointMax.y() = ( line( 0 ) * pointMax.x() + line( 2 ) ) / ( -line( 1 ) );
        cv::line( curImgBGR, cv::Point2d( pointMin.x(), pointMin.y() ), cv::Point2d( pointMax.x(), pointMax.y() ), colors.at( "amber" ) );

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

void visualization::epipolarLinesWithPoints(
  const Frame& refFrame, const Frame& curFrame, const Eigen::MatrixXd& points, const double sigma, const std::string& windowsName )
{
    cv::Mat refImgBGR;
    cv::cvtColor( refFrame.m_imagePyramid.getBaseImage(), refImgBGR, cv::COLOR_GRAY2BGR );
    cv::Mat curImgBGR;
    cv::cvtColor( curFrame.m_imagePyramid.getBaseImage(), curImgBGR, cv::COLOR_GRAY2BGR );
    // const Sophus::SE3d T_Pre2Cur = refFrame.m_TransW2F.inverse() * curFrame.m_TransW2F;
    // const Eigen::Matrix3d E = T_Pre2Cur.rotationMatrix() * algorithm::hat(T_Pre2Cur.translation());
    // std::cout << "E matrix: " << E.format( utils::eigenFormat() ) << std::endl;
    // const Eigen::Matrix3d F = refFrame.m_camera->invK().transpose() * E * curFrame.m_camera->invK();
    // std::cout << "F matrix: " << F.format( utils::eigenFormat() ) << std::endl;
    double minDepth         = 0.0;
    double maxDepth         = 0.0;
    const Eigen::Vector2d C = curFrame.camera2image( curFrame.cameraInWorld() );

    auto szPoints = refFrame.numberObservation();
    for ( std::size_t i( 0 ); i < szPoints; i++ )
    {
        const auto& reFeature = refFrame.m_frameFeatures[ i ]->m_feature;
        cv::circle( refImgBGR, cv::Point2d( reFeature.x(), reFeature.y() ), 5.0, colors.at( "pink" ) );

        // Eigen::Vector3d line = F * Eigen::Vector3d( reFeature.x(), reFeature.y(), 1.0 );
        // double nu            = line( 0 ) * line( 0 ) + line( 1 ) * line( 1 );
        // nu                   = 1 / std::sqrt( nu );
        // line *= nu;

        // const auto& normalizedVec = refFrame.m_frameFeatures[ i ]->m_bearingVec;
        const auto& point = refFrame.m_frameFeatures[ i ]->m_point->m_position;
        // std::cout << "idx " << i << ", pos: " << point.transpose() << std::endl;
        // const double curDepth = point.z();
        // minDepth                  = curDepth - sigma;
        // maxDepth                  = curDepth + sigma;
        // std::cout << "T_Pre2Cur: " << T_Pre2Cur.params().transpose() << std::endl;
        // Eigen::Vector2d pointMin = curFrame.camera2image( normalizedVec * minDepth );
        // Eigen::Vector2d pointMax = curFrame.camera2image( normalizedVec * maxDepth );
        // std::cout << "Position point 1: " << pointMin.transpose() << std::endl;
        // std::cout << "Position point 2: " << pointMax.transpose() << std::endl;
        // cv::line( curImgBGR, cv::Point2d( pointMin.x(), pointMin.y() ), cv::Point2d( pointMax.x(), pointMax.y() ),
        //   colors.at( "amber" ) );
        const Eigen::Vector2d pointCenter = curFrame.world2image( point );
        cv::circle( curImgBGR, cv::Point2d( pointCenter.x(), pointCenter.y() ), 5.0, colors.at( "orange" ) );

        // pointMin.x() = pointCenter.x() - sigma;
        // pointMin.y() = (line(0) * pointMin.x() + line(2))/(-line(1));

        // pointMax.x() = pointCenter.x() + sigma;
        // pointMax.y() = (line(0) * pointMax.x() + line(2))/(-line(1));
        // cv::line( curImgBGR, cv::Point2d( pointMin.x(), pointMin.y() ), cv::Point2d( pointMax.x(), pointMax.y() ),
        //   colors.at( "amber" ) );

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

void visualization::imagePatches( const cv::Mat& patches,
                                  const uint32_t numberPatches,
                                  const uint32_t patchSize,
                                  const uint32_t horizontalMargin,
                                  const uint32_t verticalMargin,
                                  const uint32_t maxPatchInRow )
{
    const uint32_t numberNecessaryCols = numberPatches > maxPatchInRow ? maxPatchInRow : numberPatches;
    const uint32_t numberNecessaryRows = static_cast< uint32_t >( std::ceil( numberPatches / static_cast< float >( maxPatchInRow ) ) );
    const uint32_t patchArea           = patchSize * patchSize;
    const uint32_t rows                = numberNecessaryRows * ( horizontalMargin + patchSize + 1 );
    const uint32_t cols                = numberNecessaryCols * ( verticalMargin + patchSize + 1 );

    cv::Mat outputImg( rows, cols, patches.type(), cv::Scalar( 0 ) );
    uint32_t cntPatches = 0;
    for ( std::size_t i( 0 ); i < numberNecessaryRows; i++ )
    {
        for ( std::size_t j( 0 ); j < numberNecessaryCols; j++, cntPatches++ )
        {
            if ( cntPatches == numberPatches )
                break;

            const uint32_t leftUpCornerX = j * ( horizontalMargin + patchSize ) + horizontalMargin;
            const uint32_t leftUpCornerY = i * ( verticalMargin + patchSize ) + verticalMargin;
            const cv::Mat patchContent   = patches.row( cntPatches ).reshape( 1, patchSize );
            // std::cout << "patchContent: " << patchContent << std::endl;
            auto ROI = outputImg( cv::Rect( leftUpCornerX, leftUpCornerY, patchSize, patchSize ) );
            patchContent.copyTo( ROI );
        }
    }
    // std::cout << "output: " << outputImg(cv::Rect(0, 0, 35, 35)) << std::endl;
    cv::Mat visPatches;
    // if the output image type is CV_32F, the image should be normalized between 0 and 1
    // and if the output image type is CV_8U, the image should be normalized between 0 and 255
    cv::normalize( outputImg, visPatches, 0, 1, cv::NORM_MINMAX, CV_32F );
    cv::imshow( "refPatches", visPatches );
}

cv::Scalar visualization::generateColor( const float min, const float max, const float value )
{
    uint8_t hue = static_cast< uint8_t >( ( 120.f / ( max - min ) ) * value );
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

void visualization::drawHistogram(
  std::vector< float >& vec, cv::Mat& imgHistogram, int numBins, int imageWidth, int imageHeight )
{
    // Compute and plot histogram of 1D data
    // int numBins = 50;
    auto extrema   = std::minmax_element( vec.begin(), vec.end() );
    float minValue = *extrema.first;
    float maxValue = *extrema.second;

    // OpenCV's calcHist() looks like overkill, but it is pretty fast...
    int nbins             = numBins;
    int nimages           = 1;
    int channels[]        = {0};
    int dims              = 1;
    int histSize[]        = {nbins};               // Number of bins
    float range[]         = {minValue, maxValue};  // [lower bound, upper bound[
    const float* ranges[] = {range};               // Vector of bin boundaries or single entry for uniform range
    bool uniform          = true;                  // See ranges above
    bool accumulate       = false;                 // Retain accumulator array (for iterative updates)
    const cv::Mat dataMat( vec );                  // No copy
    cv::Mat hist;                                  // 32FC1 1xN (rows x cols) for uniform range
    cv::calcHist( &dataMat, nimages, channels, cv::Mat(), hist, dims, histSize, ranges, uniform, accumulate );

    double maxCount = 0;
    cv::minMaxLoc( hist, NULL, &maxCount, NULL, NULL );
    // float binWidth = (maxValue - minValue) / nbins;

    // Draw histogram using cv::fillPoly()
    // int imageWidth = 500;
    // int imageHeight = static_cast<int>(0.61803398875 * imageWidth);
    int marginTop    = static_cast< int >( 0.1 * imageHeight );
    int marginBottom = static_cast< int >( 0.1 * imageHeight );
    float stepSize   = imageWidth / static_cast< float >( numBins - 1 );
    float scale      = ( imageHeight - marginTop - marginBottom ) / maxCount;
    cv::Scalar color( 204, 104, 0 );
    int lineType = cv::LINE_8;
    imgHistogram.create( imageHeight, imageWidth, CV_8UC3 );
    imgHistogram.setTo( cv::Scalar( 64, 64, 64 ) );

    std::vector< std::vector< cv::Point > > pts( 1 );
    std::vector< cv::Point >& polyline = pts.back();
    polyline.push_back( cv::Point( 0, imageHeight - marginBottom ) );
    for ( int i = 0; i < numBins; i++ )
    {
        int x = cvRound( stepSize * ( i ) );
        int y = imageHeight - marginBottom - static_cast< int >( scale * cvRound( hist.at< float >( i ) ) );
        polyline.push_back( cv::Point( x, y ) );
    }
    polyline.push_back( cv::Point( stepSize * ( numBins ), imageHeight - marginBottom ) );
    cv::fillPoly( imgHistogram, pts, color, lineType );

    // Draw labels
    {
        cv::Point pt1( 0, imageHeight - marginBottom );            // top left
        cv::Point pt2( pt1.x + imageWidth, pt1.y + imageHeight );  // Bottom right
        cv::Scalar color( 32, 32, 32 );
        int thickness = cv::FILLED;
        int lineType  = cv::LINE_8;
        cv::rectangle( imgHistogram, pt1, pt2, color, thickness, lineType );
    }

    {
        // std::string text = PKutils::string_printf( "%5.2f", minValue );
        std::string text;
        cv::Point origin = cv::Point( 0, imageHeight );
        int fontFace     = cv::FONT_HERSHEY_SIMPLEX;
        double fontScale = 0.75;
        cv::Scalar color( 255, 255, 255 );
        int thickness         = 1;
        int lineType          = cv::LINE_8;
        bool bottomLeftOrigin = false;

        // Calculate final width, height and baseline of text box
        int baseline;
        cv::getTextSize( text, fontFace, fontScale, thickness, &baseline );
        origin.y -= baseline;
        cv::putText( imgHistogram, text, origin, fontFace, fontScale, color, thickness, lineType, bottomLeftOrigin );
    }

    {
        // std::string text = PKutils::string_printf( "%5.2f", maxValue );
        std::string text;
        cv::Point origin = cv::Point( imageWidth, imageHeight );
        int fontFace     = cv::FONT_HERSHEY_SIMPLEX;
        double fontScale = 0.75;
        cv::Scalar color( 255, 255, 255 );
        int thickness         = 1;
        int lineType          = cv::LINE_8;
        bool bottomLeftOrigin = false;

        // Calculate final width, height and baseline of text box
        int baseline;
        cv::Size textSize = cv::getTextSize( text, fontFace, fontScale, thickness, &baseline );
        origin.x -= textSize.width;
        origin.y -= baseline;
        cv::putText( imgHistogram, text, origin, fontFace, fontScale, color, thickness, lineType, bottomLeftOrigin );
    }

    // // Draw vertical lines in histogram
    // {
    //     // Lower and upper bounds
    //     float threshLow  = medianCom - threshSigma * madCom;
    //     float threshHigh = medianCom + threshSigma * madCom;
    //     auto extrema     = std::minmax_element( histOri.begin(), histOri.end() );
    //     float minValue   = *extrema.first;
    //     float maxValue   = *extrema.second;
    //     float threshLowImage =
    //         PKutils::mapLinearInterval( threshLow, minValue, maxValue, 0.0f, static_cast< float >( imageWidth ) );
    //     float threshHighImage =
    //         PKutils::mapLinearInterval( threshHigh, minValue, maxValue, 0.0f, static_cast< float >( imageWidth ) );
    //     cv::Point threshLowTop( cvRound( threshLowImage ), 0 );
    //     cv::Point threshLowBottom( cvRound( threshLowImage ), imageHeight );
    //     cv::line( oriHistogram, threshLowTop, threshLowBottom, cv::Scalar( 255, 255, 255 ), 1, cv::LINE_8 );
    //     cv::Point threshHighTop( cvRound( threshHighImage ), 0 );
    //     cv::Point threshHighBottom( cvRound( threshHighImage ), imageHeight );
    //     cv::line( oriHistogram, threshHighTop, threshHighBottom, cv::Scalar( 255, 255, 255 ), 1, cv::LINE_8 );
    // }
}