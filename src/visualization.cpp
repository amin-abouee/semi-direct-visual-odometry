#include "visualization.hpp"
#include "algorithm.hpp"
#include "feature.hpp"
#include "utils.hpp"

#include <unordered_map>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "easylogging++.h"
#define Visualization_Log( LEVEL ) CLOG( LEVEL, "Visualization" )

cv::Mat visualization::getGrayImage( const cv::Mat& img )
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

    cv::Mat imgGray;
    // inputImage is color
    if ( img.channels() == 3 )
    {
        cv::cvtColor( img, imgGray, cv::COLOR_BGR2GRAY );
    }
    cv::normalize( imgGray, imgGray, 0, 255, cv::NORM_MINMAX, CV_8UC1 );
    return imgGray;
}

cv::Mat visualization::getColorImage( const cv::Mat& img )
{
    cv::Mat imgBGR;
    if ( img.channels() == 1 )
        cv::cvtColor( img, imgBGR, cv::COLOR_GRAY2BGR );
    else
        imgBGR = img.clone();
    return imgBGR;
}

cv::Mat visualization::getHSVImageWithMagnitude( const cv::Mat& img, const uint8_t minMagnitude )
{
    // https://realpython.com/python-opencv-color-spaces/
    // https://stackoverflow.com/questions/23001512/c-and-opencv-get-and-set-pixel-color-to-mat
    // https://answers.opencv.org/question/178766/adjust-hue-and-saturation-like-photoshop/
    // http://colorizer.org/
    // https://toolstud.io/color/rgb.php?rgb_r=0&rgb_g=255&rgb_b=0&convert=rgbdec
    // https://answers.opencv.org/question/191488/create-a-hsv-range-palette/

    // https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html

    cv::Mat normMag, imgBGR, imgHSV;
    // check type of image, gray or color
    if ( img.channels() == 1 )
    {
        cv::normalize( img, img, 0, 255, cv::NORM_MINMAX, CV_8UC1 );
        cv::cvtColor( img, imgBGR, cv::COLOR_GRAY2BGR );
        cv::cvtColor( imgBGR, imgHSV, cv::COLOR_BGR2HSV );
    }
    else
    {
        cv::cvtColor( img, imgHSV, cv::COLOR_BGR2HSV );
    }

    for ( int32_t i( 0 ); i < imgHSV.rows; i++ )
    {
        for ( int32_t j( 0 ); j < imgHSV.cols; j++ )
        {
            if ( img.at< uint8_t >( i, j ) >= minMagnitude )
            {
                cv::Vec3b& px    = imgHSV.at< cv::Vec3b >( cv::Point( j, i ) );
                cv::Scalar color = generateColor( minMagnitude, 255, img.at< uint8_t >( i, j ) );
                px[ 0 ]          = static_cast< uint8_t >( color[ 0 ] );
                px[ 1 ]          = static_cast< uint8_t >( color[ 1 ] );
                px[ 2 ]          = static_cast< uint8_t >( color[ 2 ] );
            }
        }
    }

    cv::cvtColor( imgHSV, imgBGR, cv::COLOR_HSV2BGR );
    return imgBGR;
}

cv::Scalar visualization::generateColor( const uint8_t min, const uint8_t max, const uint8_t value )
{
    const int32_t diff = value - min;
    const double hue   = diff * ( 120.0 / ( max - min ) );
    return cv::Scalar( hue, 100.0, 100.0 );
}

void visualization::stickTwoImageVertically( const cv::Mat& refImg, const cv::Mat& curImg, cv::Mat& img )
{
    cv::vconcat( refImg, curImg, img );
}

void visualization::stickTwoImageHorizontally( const cv::Mat& refImg, const cv::Mat& curImg, cv::Mat& img )
{
    cv::hconcat( refImg, curImg, img );
}

void visualization::featurePoints(
  cv::Mat& img,
  const std::shared_ptr< Frame >& frame,
  const u_int32_t radiusSize,
  const std::string& color,
  const std::function< void( cv::Mat& img, const Eigen::Vector2d& point, const u_int32_t size, const cv::Scalar& color ) >& drawingFunctor )
{
    cv::Scalar colorRGB;
    if ( colors.find( color ) != colors.end() )
        colorRGB = colors.at( color );
    else
        colorRGB = colors.at( "white" );

    const auto szPoints = frame->numberObservation();
    int32_t cnt = 0;
    for ( std::size_t i( 0 ); i < szPoints; i++ )
    {
        const auto& feature = frame->m_features[ i ];
        if ( feature->m_point != nullptr )
        {
            drawingFunctor( img, feature->m_pixelPosition, radiusSize, colorRGB );
            cnt++;
        }
        else
        {
            drawingFunctor( img, feature->m_pixelPosition, radiusSize, colors.at( "cyan" ) );
        }
    }
    Visualization_Log(WARNING) << "(featurePoints) number of points: " << cnt;

}

void visualization::imageGrid( cv::Mat& img, const int32_t gridSize, const std::string& color )
{
    cv::Scalar colorRGB;
    if ( colors.find( color ) != colors.end() )
        colorRGB = colors.at( color );
    else
        colorRGB = colors.at( "white" );

    const int width  = img.cols;
    const int height = img.rows;

    const int cols = width / gridSize;
    const int rows = height / gridSize;
    for ( int r( 1 ); r <= rows; r++ )
    {
        cv::line( img, cv::Point2i( 0, r * gridSize ), cv::Point2i( width, r * gridSize ), colorRGB );
    }

    for ( int c( 1 ); c <= cols; c++ )
    {
        cv::line( img, cv::Point2i( c * gridSize, 0 ), cv::Point2i( c * gridSize, height ), colorRGB );
    }
}

// void visualization::project3DPoints(
//   cv::Mat& img,
//   const std::shared_ptr< Frame >& frame,
//   const u_int32_t radiusSize,
//   const std::string& color,
//   const std::function< void( cv::Mat& img, const Eigen::Vector2d& point, const u_int32_t size, const cv::Scalar& color ) >&
//   drawingFunctor )
// {
//     cv::Scalar colorRGB;
//     if ( colors.find( color ) != colors.end() )
//         colorRGB = colors.at( color );
//     else
//         colorRGB = colors.at( "white" );

//     const auto szPoints = frame->numberObservation();
//     for ( std::size_t i( 0 ); i < szPoints; i++ )
//     {
//         if (frame->m_features[i]->m_point != nullptr)
//         {
//             const Eigen::Vector3d& point = frame->m_features[ i ]->m_point->m_position;
//             const auto& feature          = frame->world2image( point );
//             drawingFunctor( img, feature, radiusSize, colorRGB );
//         }
//     }
// }

void visualization::colormapDepth( cv::Mat& img,
                                   const std::shared_ptr< Frame >& frame,
                                   const u_int32_t radiusSize,
                                   const std::string& color )
{
    cv::Scalar colorRGB;
    if ( colors.find( color ) != colors.end() )
        colorRGB = colors.at( color );
    else
        colorRGB = colors.at( "white" );

    // const Eigen::Vector3d C = frame->cameraInWorld();
    const auto szPoints = frame->numberObservation();
    std::vector< double > depths;
    depths.reserve( szPoints );
    double minDepth = std::numeric_limits< double >::max();
    double maxDepth = 0.0;
    for ( std::size_t i( 0 ); i < szPoints; i++ )
    {
        const auto& feature = frame->m_features[ i ];
        if ( feature->m_point != nullptr )
        {
            const double depth = ( frame->m_absPose * feature->m_point->m_position ).z();
            depths.push_back( depth );
            minDepth = depth < minDepth ? depth : minDepth;
            maxDepth = depth > maxDepth ? depth : maxDepth;
        }
    }
    const double rangeDepth = maxDepth - minDepth;
    const double stepColor  = rangeDepth / 35.0;
    Visualization_Log( DEBUG ) << "depth min: " << minDepth << ", max: " << maxDepth << ", range: " << rangeDepth
                               << ", step color: " << stepColor;

    cv::Mat normMag, imgBGR, imgHSV;
    cv::normalize( img, img, 0, 255, cv::NORM_MINMAX, CV_8UC1 );
    cv::cvtColor( img, imgBGR, cv::COLOR_GRAY2BGR );
    cv::cvtColor( imgBGR, imgHSV, cv::COLOR_BGR2HSV );

    int32_t cnt = 0;
    for ( std::size_t i( 0 ); i < szPoints; i++ )
    {
        const auto& feature = frame->m_features[ i ];
        // Visualization_Log( DEBUG ) << "depth: " << depths[ cnt ];
        if ( feature->m_point != nullptr )
        {
            // if (depths[ cnt ] < 1.0)
            // {
            // Visualization_Log( DEBUG ) << "depth: " << depths[ cnt ];
            // cv::Vec3b& px         = imgHSV.at< cv::Vec3b >( cv::Point( feature->m_pixelPosition.y(), feature->m_pixelPosition.x() ) );
            // const double fraction = depths[ cnt ] - minDepth / ( rangeDepth );
            const int32_t fraction = depths[ cnt ] / stepColor;
            // Visualization_Log( DEBUG ) << "fraction: " << fraction;
            const uint8_t hue = static_cast< uint8_t >( fraction * 35 );
            cv::Scalar color( hue, 100, 100 );
            if ( fraction == 1 )
            {
                Visualization_Log( DEBUG ) << "depth: " << depths[ cnt ] << ", hue: " << uint32_t( hue );
                cv::rectangle( imgHSV, cv::Point2d( feature->m_pixelPosition.x() - radiusSize, feature->m_pixelPosition.y() - radiusSize ),
                               cv::Point2d( feature->m_pixelPosition.x() + radiusSize, feature->m_pixelPosition.y() + radiusSize ), color,
                               2 );
            }
            // px[ 0 ]          = static_cast< uint8_t >( color[ 0 ] );
            // px[ 1 ]          = static_cast< uint8_t >( color[ 1 ] );
            // px[ 2 ]          = static_cast< uint8_t >( color[ 2 ] );
            // }
            // }
            cnt++;
        }
    }
    cv::cvtColor( imgHSV, imgBGR, cv::COLOR_HSV2BGR );
    img = imgBGR;
}

void visualization::projectPointsWithRelativePose(
  cv::Mat& img,
  const std::shared_ptr< Frame >& refFrame,
  const std::shared_ptr< Frame >& curFrame,
  const u_int32_t radiusSize,
  const std::string& color,
  const std::function< void( cv::Mat& img, const Eigen::Vector2d& point, const u_int32_t size, const cv::Scalar& color ) >& drawingFunctor )
{
    const Sophus::SE3d relativePose = algorithm::computeRelativePose( refFrame, curFrame );

    const auto szPoints     = refFrame->numberObservation();
    const Eigen::Vector3d C = refFrame->cameraInWorld();
    int32_t cnt = 0;
    for ( std::size_t i( 0 ); i < szPoints; i++ )
    {
        const auto& feature = refFrame->m_features[ i ];
        if ( feature->m_point != nullptr )
        {
            const double depthNorm = ( feature->m_point->m_position - C ).norm();
            const Eigen::Vector3d refPoint( feature->m_bearingVec * depthNorm );
            const Eigen::Vector3d curPoint( relativePose * refPoint );
            const Eigen::Vector2d curFeature( curFrame->camera2image( curPoint ) );
            drawingFunctor( img, curFeature, radiusSize, colors.at( color ) );
            cnt++;
        }
    }
    Visualization_Log(WARNING) << "(projectPointsWithRelativePose) number of points: " << cnt;
}

void visualization::projectLinesWithRelativePose(
  cv::Mat& img,
  const std::shared_ptr< Frame >& refFrame,
  const std::shared_ptr< Frame >& curFrame,
  const uint32_t rangeInPixels,
  const std::string& color,
  const std::function< void( cv::Mat& img, const Eigen::Vector2d& point1, const Eigen::Vector2d& point2, const cv::Scalar& color ) >&
    drawingFunctor )
{
    // const Sophus::SE3d relativePose = refFrame->m_absPose.inverse() * curFrame->m_absPose;

    const Sophus::SE3d relativePose = algorithm::computeRelativePose( refFrame, curFrame );
    const Eigen::Matrix3d E         = relativePose.rotationMatrix() * algorithm::hat( relativePose.translation() );
    const Eigen::Matrix3d F         = refFrame->m_camera->invK().transpose() * E * curFrame->m_camera->invK();
    projectLinesWithF( img, refFrame, F, rangeInPixels, color, drawingFunctor );
}

void visualization::projectLinesWithF(
  cv::Mat& img,
  const std::shared_ptr< Frame >& refFrame,
  const Eigen::Matrix3d& F,
  const uint32_t rangeInPixels,
  const std::string& color,
  const std::function< void( cv::Mat& img, const Eigen::Vector2d& point1, const Eigen::Vector2d& point2, const cv::Scalar& color ) >&
    drawingFunctor )
{
    cv::Scalar colorRGB;
    if ( colors.find( color ) != colors.end() )
        colorRGB = colors.at( color );
    else
        colorRGB = colors.at( "white" );

    const auto szPoints = refFrame->numberObservation();
    for ( std::size_t i( 0 ); i < szPoints; i++ )
    {
        const auto& refHomogenous = refFrame->m_features[ i ]->m_homogenous;
        Eigen::Vector3d line      = F * refHomogenous;
        double nu                 = line( 0 ) * line( 0 ) + line( 1 ) * line( 1 );
        nu                        = 1 / std::sqrt( nu );
        line *= nu;

        const Eigen::Vector3d pointCenter = line / line( 2 );
        Eigen::Vector2d pointMin;
        Eigen::Vector2d pointMax;
        pointMin.x() = pointCenter.x() - rangeInPixels;
        pointMin.y() = ( line( 0 ) * pointMin.x() + line( 2 ) ) / ( -line( 1 ) );

        pointMax.x() = pointCenter.x() + rangeInPixels;
        pointMax.y() = ( line( 0 ) * pointMax.x() + line( 2 ) ) / ( -line( 1 ) );
        // cv::line( img, cv::Point2d( pointMin.x(), pointMin.y() ), cv::Point2d( pointMax.x(), pointMax.y() ), colors.at( "amber" ) );
        drawingFunctor( img, pointMin, pointMax, colorRGB );
    }
}

void visualization::epipole(
  cv::Mat& img,
  const std::shared_ptr< Frame >& frame,
  const u_int32_t radiusSize,
  const std::string& color,
  const std::function< void( cv::Mat& img, const Eigen::Vector2d& point, const u_int32_t size, const cv::Scalar& color ) >& drawingFunctor )
{
    cv::Scalar colorRGB;
    if ( colors.find( color ) != colors.end() )
        colorRGB = colors.at( color );
    else
        colorRGB = colors.at( "white" );

    const Eigen::Vector2d C = frame->camera2image( frame->cameraInWorld() );
    drawingFunctor( img, C, radiusSize, colorRGB );
}

cv::Mat visualization::referencePatches( const cv::Mat& patches,
                                     const uint32_t numberPatches,
                                     const uint32_t patchSize,
                                     const uint32_t horizontalMargin,
                                     const uint32_t verticalMargin,
                                     const uint32_t maxPatchInRow )
{
    const uint32_t numberNecessaryCols = numberPatches > maxPatchInRow ? maxPatchInRow : numberPatches;
    const uint32_t numberNecessaryRows = static_cast< uint32_t >( std::ceil( numberPatches / static_cast< float >( maxPatchInRow ) ) );
    // const uint32_t patchArea           = patchSize * patchSize;
    const uint32_t rows = numberNecessaryRows * ( horizontalMargin + patchSize + 1 );
    const uint32_t cols = numberNecessaryCols * ( verticalMargin + patchSize + 1 );

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
    // cv::imshow( "refPatches", visPatches );
    return visPatches;
}

cv::Mat visualization::residualsPatches( const Eigen::VectorXd& residuals,
                                         const uint32_t numberPatches,
                                         const uint32_t patchSize,
                                         const uint32_t horizontalMargin,
                                         const uint32_t verticalMargin,
                                         const uint32_t maxPatchInRow )
{
    const uint32_t numberNecessaryCols = numberPatches > maxPatchInRow ? maxPatchInRow : numberPatches;
    const uint32_t numberNecessaryRows = static_cast< uint32_t >( std::ceil( numberPatches / static_cast< float >( maxPatchInRow ) ) );
    const uint32_t patchArea           = patchSize * patchSize;
    const uint32_t rows                = numberNecessaryRows * ( horizontalMargin + patchSize ) + horizontalMargin;
    const uint32_t cols                = numberNecessaryCols * ( verticalMargin + patchSize ) + verticalMargin;
    // int rowsSz = patchSize;

    cv::Mat outputImg( rows, cols, CV_8U, cv::Scalar( 0 ) );
    uint32_t cntPatches = 0;
    for ( std::size_t i( 0 ); i < numberNecessaryRows; i++ )
    {
        for ( std::size_t j( 0 ); j < numberNecessaryCols; j++, cntPatches++ )
        {
            if ( cntPatches == numberPatches )
                break;

            const uint32_t leftUpCornerX = j * ( horizontalMargin + patchSize ) + horizontalMargin;
            const uint32_t leftUpCornerY = i * ( verticalMargin + patchSize ) + verticalMargin;
            // Mat (int rows, int cols, int type, void *data, size_t step=AUTO_STEP)
            double* data = const_cast< double* >( &residuals( cntPatches * patchArea ) );
            cv::Mat patchContent( patchSize, patchSize, CV_64F, data );
            patchContent = cv::abs( patchContent );
            double min, max;
            cv::minMaxLoc( patchContent, &min, &max );
            // patchContent.convertTo(patchContent, CV_32F);
            // std::cout << "type: " << patchContent.type() << std::endl;
            // std::cout << patchContent << std::endl;
            cv::Mat tmpPatch;
            cv::normalize( patchContent, tmpPatch, 0, max, cv::NORM_MINMAX, CV_8U );
            // std::cout << "patch content type: " << tmpPatch.type() << std::endl;
            // const cv::Mat patchContent   = patches.row( cntPatches ).reshape( 1, patchSize );
            // std::cout << "patchContent: " << patchContent << std::endl;
            auto ROI = outputImg( cv::Rect( leftUpCornerX, leftUpCornerY, patchSize, patchSize ) );
            tmpPatch.copyTo( ROI );
        }
    }
    // std::cout << "output: " << outputImg(cv::Rect(0, 0, 35, 35)) << std::endl;
    // cv::Mat visPatches;
    // if the output image type is CV_32F, the image should be normalized between 0 and 1
    // and if the output image type is CV_8U, the image should be normalized between 0 and 255
    // cv::normalize( outputImg, visPatches, 0, 255, cv::NORM_MINMAX, CV_8U );
    // return visPatches;
    return outputImg;
}

void visualization::drawHistogram( std::map< std::string, std::any >& pack )
{
    // https://www.tutorialspoint.com/matplotlib/matplotlib_subplot2grid_function.htm
    auto height = std::any_cast< uint32_t >( pack[ "figure_size_height" ] );
    auto width  = std::any_cast< uint32_t >( pack[ "figure_size_width" ] );
    plt::figure_size( width, height );

    try
    {
        auto residuals      = std::any_cast< std::vector< double > >( pack[ "residuals_data" ] );
        auto residualsColor = std::any_cast< std::string >( pack[ "residuals_color" ] );
        auto numberBins     = std::any_cast< uint32_t >( pack[ "residuals_number_bins" ] );
        auto median         = std::any_cast< double >( pack[ "residuals_median" ] );
        auto medianColor    = std::any_cast< std::string >( pack[ "residuals_median_color" ] );
        auto sigma          = std::any_cast< double >( pack[ "residuals_sigma" ] );
        auto sigmaColor     = std::any_cast< std::string >( pack[ "residuals_sigma_color" ] );
        auto mad            = std::any_cast< double >( pack[ "residuals_mad" ] );
        auto madColor       = std::any_cast< std::string >( pack[ "residuals_mad_color" ] );
        auto windowsName    = std::any_cast< std::string >( pack[ "residuals_windows_name" ] );

        const auto max = *std::max_element( residuals.begin(), residuals.end() );
        const auto min = *std::min_element( residuals.begin(), residuals.end() );
        std::vector< uint32_t > binsValue( numberBins, 0 );
        const double rangeBins = ( std::ceil( max ) - std::floor( min ) ) / numberBins;
        // std::cout << "max val: " << max << ", min: " << min << ", range bin: " << rangeBins << std::endl;

        for ( std::size_t i( 0 ); i < residuals.size(); i++ )
        {
            const uint32_t idx = ( residuals[ i ] - min ) / rangeBins;
            binsValue[ idx ]++;
        }
        // for(std::size_t i(0); i<binsValue.size(); i++)
        // {
        //     std::cout << "id: " << i << " size: " << binsValue[i] << std::endl;
        // }

        const double maxBins        = *std::max_element( binsValue.begin(), binsValue.end() );
        const uint32_t numberSample = 30;
        // std::cout << "max bins: " << maxBins << std::endl;
        std::vector< double > y;
        std::vector< double > medVec( numberSample, median );
        std::vector< double > minusSigmaVec( numberSample, median + sigma );
        std::vector< double > plusSigmaVec( numberSample, median - sigma );
        std::vector< double > minusMADVec( numberSample, median - mad );
        std::vector< double > plusMADVec( numberSample, median + mad );

        std::vector< double > yRugPlot( residuals.size(), -1.0 );

        for ( uint32_t i( 0 ); i < numberSample; i++ )
        {
            y.push_back( ( maxBins / numberSample ) * i );
        }

        // std::cout << "res: " << residuals.size() << ", rug: " << yRugPlot.size() << std::endl;
        std::map< std::string, std::string > keywords;
        keywords[ "zorder" ] = "100";
        plt::subplot2grid( 9, 11, 0, 0, 4, 6 );
        keywords[ "c" ] = medianColor;
        plt::scatter( medVec, y, 1.0, keywords );

        keywords[ "c" ] = madColor;
        plt::scatter( minusMADVec, y, 1.0, keywords );
        plt::scatter( plusMADVec, y, 1.0, keywords );

        keywords[ "c" ] = sigmaColor;
        plt::scatter( minusSigmaVec, y, 1.0, keywords );
        plt::scatter( plusSigmaVec, y, 1.0, keywords );

        std::map< std::string, std::string > keywords_map;
        keywords_map[ "marker" ] = "|";
        keywords_map[ "ls" ]     = "";
        // keywords_map[ "markersize" ] = "10";
        keywords_map[ "color" ] = "mediumseagreen";
        // keywords_map[ "zorder" ] = "100";
        plt::plot( residuals, yRugPlot, keywords_map );

        plt::hist( residuals, numberBins, residualsColor );
        // plt::legend();
        plt::xlabel( windowsName );
        plt::ylabel( "numbers" );
        plt::title( windowsName );
    }
    catch ( const std::bad_any_cast& e )
    {
        std::cerr << e.what() << '\n';
    }

    try
    {
        // pack["weights_data"] = weights;
        // pack["weights_windows_name"] = "weights";
        // pack["weights_color"] = "green";

        auto weights     = std::any_cast< std::vector< double > >( pack[ "weights_data" ] );
        auto numberBins  = std::any_cast< uint32_t >( pack[ "weights_number_bins" ] );
        auto color       = std::any_cast< std::string >( pack[ "weights_color" ] );
        auto windowsName = std::any_cast< std::string >( pack[ "weights_windows_name" ] );

        plt::subplot2grid( 9, 11, 5, 0, 4, 6 );
        plt::hist( weights, numberBins, color );
        plt::xlabel( windowsName );
        plt::ylabel( "numbers" );
        plt::title( windowsName );
    }
    catch ( const std::bad_any_cast& e )
    {
        std::cerr << e.what() << '\n';
    }

    try
    {
        // pack["patches_cv"] = resPatches;
        // pack["patche_windows_name"] = "patche";
        // pack["patches_colormap"] = std::string("viridis_r");

        auto patches     = std::any_cast< cv::Mat >( pack[ "patches_cv" ] );
        auto colormap    = std::any_cast< std::string >( pack[ "patches_colormap" ] );
        auto windowsName = std::any_cast< std::string >( pack[ "patches_windows_name" ] );

        std::map< std::string, std::string > keywords;
        plt::subplot2grid( 9, 11, 0, 7, 4, 4 );
        keywords[ "cmap" ] = colormap;
        plt::imshow( patches.ptr(), patches.rows, patches.cols, 1, keywords );
        plt::title( windowsName );
    }
    catch ( const std::bad_any_cast& e )
    {
        std::cerr << e.what() << '\n';
    }

    try
    {
        // pack["hessian_cv"] = cvHessianGray;
        // pack["hessian_windows_name"] = "hessian";

        auto hessian     = std::any_cast< cv::Mat >( pack[ "hessian_cv" ] );
        auto colormap    = std::any_cast< std::string >( pack[ "hessian_colormap" ] );
        auto windowsName = std::any_cast< std::string >( pack[ "hessian_windows_name" ] );

        std::map< std::string, std::string > keywords;

        // https://answers.opencv.org/question/27248/max-and-min-values-in-a-mat/
        // double min, max;
        // cv::minMaxLoc( hessian, &min, &max );
        // const float maxAbsolute = std::max( std::abs( min ), std::abs( max ) );
        // std::cout << "Min: " << min << ", Abs Max: " << max << std::endl;
        // std::cout << "Abs Min: " << -maxAbsolute << ", Abs Max: " << maxAbsolute << std::endl;
        // https://matplotlib.org/tutorials/colors/colormaps.html#diverging
        plt::subplot2grid( 9, 11, 5, 7, 4, 4 );
        keywords[ "cmap" ] = colormap;
        // keywords["vmin"] = std::to_string(-1.0);
        // keywords["vmax"] = std::to_string(1.0);
        // std::cout << "vmin: " << keywords["vmin"] << ", vmax: " << keywords["vmax"] << std::endl;
        plt::imshow( hessian.ptr(), hessian.rows, hessian.cols, 1, keywords );
        plt::title( windowsName );
    }
    catch ( const std::bad_any_cast& e )
    {
        std::cerr << e.what() << '\n';
    }

    plt::show();
}

void visualization::projectDepthFilters(
  cv::Mat& img,
  const std::shared_ptr< Frame >& frame,
  const std::vector< MixedGaussianFilter >& depthFilters,
  const u_int32_t radiusSize,
  const std::string& color,
  const std::function< void( cv::Mat& img, const Eigen::Vector2d& point1, const Eigen::Vector2d& point2, const cv::Scalar& color ) >&
    drawingFunctor )
{
    // const uint32_t imgWidth = frame->m_camera->width();
    for ( auto& depthFilter : depthFilters )
    {
        const Sophus::SE3d relativePose        = algorithm::computeRelativePose( depthFilter.m_feature->m_frame, frame );
        const Eigen::Vector3d pointInCurCamera = relativePose * ( depthFilter.m_feature->m_bearingVec / depthFilter.m_mu );
        const Eigen::Vector2d pointInCurImage  = frame->camera2image( pointInCurCamera );
        if ( pointInCurCamera.z() < 0 || frame->m_camera->isInFrame( pointInCurImage ) == false )
        {
            continue;
        }

        // cv::circle( img, cv::Point2d( depthFilter.m_feature->m_pixelPosition.x(), depthFilter.m_feature->m_pixelPosition.y() ),
        // radiusSize, colors.at( "pink" ) );
        cv::circle( img, cv::Point2d( pointInCurImage.x(), pointInCurImage.y() ), radiusSize, colors.at( "orange" ) );

        // inverse representation of depth
        const double inverseMinDepth = depthFilter.m_mu + depthFilter.m_var;
        const double inverseMaxDepth = std::max( depthFilter.m_mu - depthFilter.m_var, 1e-7 );

        const Eigen::Vector2d projectedMinDepth =
          frame->camera2image( relativePose * ( depthFilter.m_feature->m_bearingVec / inverseMinDepth ) );
        const Eigen::Vector2d projectedMaxDepth =
          frame->camera2image( relativePose * ( depthFilter.m_feature->m_bearingVec / inverseMaxDepth ) );

        drawingFunctor( img, Eigen::Vector2d( projectedMinDepth.x(), projectedMinDepth.y() ),
                        Eigen::Vector2d( projectedMaxDepth.x(), projectedMaxDepth.y() ), colors.at( color ) );
    }
}

void visualization::projectDepthFilters(
  cv::Mat& img,
  const std::shared_ptr< Frame >& frame,
  const std::vector< MixedGaussianFilter >& depthFilters,
  const std::vector< double >& updatedDepths,
  const u_int32_t radiusSize,
  const std::string& color,
  const std::function< void( cv::Mat& img, const Eigen::Vector2d& point1, const Eigen::Vector2d& point2, const cv::Scalar& color ) >&
    drawingFunctor )
{
    // const uint32_t imgWidth = frame->m_camera->width();
    std::cout << "updatedDepths.size(): " << updatedDepths.size() << ", depthFilters.size(): " << depthFilters.size() << std::endl;
    assert( updatedDepths.size() == depthFilters.size() );
    uint32_t idx = 0;
    for ( auto& depthFilter : depthFilters )
    {
        const Sophus::SE3d relativePose        = algorithm::computeRelativePose( depthFilter.m_feature->m_frame, frame );
        const Eigen::Vector3d pointInCurCamera = relativePose * ( depthFilter.m_feature->m_bearingVec / depthFilter.m_mu );
        const Eigen::Vector2d pointInCurImage  = frame->camera2image( pointInCurCamera );
        if ( pointInCurCamera.z() < 0 || frame->m_camera->isInFrame( pointInCurImage ) == false )
        {
            idx++;
            continue;
        }

        // cv::circle( img, cv::Point2d( depthFilter.m_feature->m_pixelPosition.x(), depthFilter.m_feature->m_pixelPosition.y() ),
        // radiusSize, colors.at( "pink" ) );
        cv::circle( img, cv::Point2d( pointInCurImage.x(), pointInCurImage.y() ), radiusSize, colors.at( "orange" ) );

        const Eigen::Vector2d updatedLocationInCurImage =
          frame->camera2image( relativePose * ( depthFilter.m_feature->m_bearingVec * updatedDepths[ idx++ ] ) );
        cv::circle( img, cv::Point2d( updatedLocationInCurImage.x(), updatedLocationInCurImage.y() ), radiusSize, colors.at( "purple" ) );

        // inverse representation of depth
        const double inverseMinDepth = depthFilter.m_mu + depthFilter.m_var;
        const double inverseMaxDepth = std::max( depthFilter.m_mu - depthFilter.m_var, 1e-7 );

        const Eigen::Vector2d projectedMinDepth =
          frame->camera2image( relativePose * ( depthFilter.m_feature->m_bearingVec / inverseMinDepth ) );
        const Eigen::Vector2d projectedMaxDepth =
          frame->camera2image( relativePose * ( depthFilter.m_feature->m_bearingVec / inverseMaxDepth ) );

        drawingFunctor( img, Eigen::Vector2d( projectedMinDepth.x(), projectedMinDepth.y() ),
                        Eigen::Vector2d( projectedMaxDepth.x(), projectedMaxDepth.y() ), colors.at( color ) );
    }
}