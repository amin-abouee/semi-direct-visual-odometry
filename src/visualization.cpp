#include "visualization.hpp"

#include <unordered_map>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "algorithm.hpp"
#include "feature.hpp"
#include "utils.hpp"

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

void visualization::templatePatches( const cv::Mat& patches,
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

void visualization::drawHistogram( std::map< std::string, std::any >& pack )
{
    // https://www.tutorialspoint.com/matplotlib/matplotlib_subplot2grid_function.htm
    auto height = std::any_cast< uint32_t >( pack[ "figure_size_height" ] );
    auto width  = std::any_cast< uint32_t >( pack[ "figure_size_width" ] );
    plt::figure_size( width, height );

    try
    {
        // pack["residuals_data"] = residuals;
        // pack["residuals_color"] = std::string("gray");
        // pack["residuals_median"] = median;
        // pack["residuals_median_color"] = "blue";
        // pack["residuals_sigma"] = sigma;
        // pack["residuals_sigma_color"] = "orange";
        // pack["residuals_mad"] = mad;
        // pack["residuals_mad_color"] = std::string("red");
        // pack["residuals_windows_name"] = std::string("residuals");

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

        for ( int i( 0 ); i < numberSample; i++ )
        {
            y.push_back( ( maxBins / numberSample ) * i );
        }

        // std::cout << "res: " << residuals.size() << ", rug: " << yRugPlot.size() << std::endl;
        std::unordered_map< std::string, std::string > keywords;
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

void visualization::featurePoints(
  cv::Mat& img,
  const Frame& frame,
  const std::function< void( cv::Mat& img, const Eigen::Vector2d& point, const u_int32_t size, const std::string& color ) >&
    drawingFunctor )
{
    const auto szPoints = frame.numberObservation();
    // std::cout << "# observation visualization: " << szPoints << std::endl;
    for ( std::size_t i( 0 ); i < szPoints; i++ )
    {
        const auto& feature = frame.m_frameFeatures[ i ]->m_feature;
        // cv::circle( img, cv::Point2d( feature.x(), feature.y() ), 5.0, colors.at( "pink" ) );
        drawingFunctor(img, feature, 11, "pink");
    }
}

void visualization::featurePointsInGrid( cv::Mat& img, const Frame& frame, const int32_t gridSize )
{
    // featurePoints( img, frame );

    const int width  = img.cols;
    const int height = img.rows;

    const int cols = width / gridSize;
    const int rows = height / gridSize;
    for ( int r( 1 ); r <= rows; r++ )
    {
        cv::line( img, cv::Point2i( 0, r * gridSize ), cv::Point2i( width, r * gridSize ), colors.at( "amber" ) );
    }

    for ( int c( 1 ); c <= cols; c++ )
    {
        cv::line( img, cv::Point2i( c * gridSize, 0 ), cv::Point2i( c * gridSize, height ), colors.at( "amber" ) );
    }
}

void visualization::project3DPoints( cv::Mat& img, const Frame& frame )
{
    const auto szPoints = frame.numberObservation();
    for ( std::size_t i( 0 ); i < szPoints; i++ )
    {
        const Eigen::Vector3d& point = frame.m_frameFeatures[ i ]->m_point->m_position;
        const auto& feature          = frame.world2image( point );
        cv::circle( img, cv::Point2d( feature.x(), feature.y() ), 8.0, colors.at( "pink" ) );
    }
}

void visualization::projectPointsWithRelativePose( cv::Mat& img, const Frame& refFrame, const Frame& curFrame )
{
    const Sophus::SE3d relativePose = refFrame.m_TransW2F.inverse() * curFrame.m_TransW2F;
    // const Eigen::Matrix3d E      = relativePose.rotationMatrix() * algorithm::hat( relativePose.translation() );
    // const Eigen::Matrix3d F = refFrame.m_camera->invK().transpose() * E * curFrame.m_camera->invK();
    // projectPointsWithF(img, refFrame, F);

    const auto szPoints     = refFrame.numberObservation();
    const Eigen::Vector3d C = refFrame.cameraInWorld();
    for ( std::size_t i( 0 ); i < szPoints; i++ )
    {
        const auto& feature    = refFrame.m_frameFeatures[ i ];
        const double depthNorm = ( feature->m_point->m_position - C ).norm();
        const Eigen::Vector3d refPoint( feature->m_bearingVec * depthNorm );
        const Eigen::Vector3d curPoint( relativePose * refPoint );
        const Eigen::Vector2d curFeature( curFrame.camera2image( curPoint ) );
        cv::circle( img, cv::Point2d( curFeature.x(), curFeature.y() ), 8.0, colors.at( "orange" ) );
    }
}

// void visualization::projectPointsWithF (cv::Mat& img, const Frame& refFrame, const Eigen::Matrix3d& F)
// {
//     const auto szPoints = refFrame.numberObservation();
//     for ( std::size_t i( 0 ); i < szPoints; i++ )
//     {
//         const auto& refHomogenous = refFrame.m_frameFeatures[ i ]->m_homogenous;
//         Eigen::Vector3d line = F * refHomogenous;
//         double nu            = line( 0 ) * line( 0 ) + line( 1 ) * line( 1 );
//         nu                   = 1 / std::sqrt( nu );
//         line *= nu;
//         line /= line(2);
//         cv::circle( img, cv::Point2d( line.x(), line.y() ), 5.0, colors.at( "orange") );
//     }
// }

void visualization::projectLinesWithRelativePose( cv::Mat& img, const Frame& refFrame, const Frame& curFrame, const uint32_t rangeInPixels )
{
    const Sophus::SE3d relativePose = refFrame.m_TransW2F.inverse() * curFrame.m_TransW2F;
    const Eigen::Matrix3d E         = relativePose.rotationMatrix() * algorithm::hat( relativePose.translation() );
    const Eigen::Matrix3d F         = refFrame.m_camera->invK().transpose() * E * curFrame.m_camera->invK();
    projectLinesWithF( img, refFrame, F, rangeInPixels );
}

void visualization::projectLinesWithF( cv::Mat& img, const Frame& refFrame, const Eigen::Matrix3d& F, const uint32_t rangeInPixels )
{
    const auto szPoints = refFrame.numberObservation();
    for ( std::size_t i( 0 ); i < szPoints; i++ )
    {
        const auto& refHomogenous = refFrame.m_frameFeatures[ i ]->m_homogenous;
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
        cv::line( img, cv::Point2d( pointMin.x(), pointMin.y() ), cv::Point2d( pointMax.x(), pointMax.y() ), colors.at( "amber" ) );
    }
}

void visualization::epipole( cv::Mat& img, const Frame& frame )
{
    const Eigen::Vector2d C = frame.camera2image( frame.cameraInWorld() );
    cv::circle( img, cv::Point2d( C.x(), C.y() ), 8.0, colors.at( "red" ) );
}

void visualization::stickTwoImageVertically( const cv::Mat& refImg, const cv::Mat& curImg, cv::Mat& img )
{
    cv::vconcat( refImg, curImg, img );
}

void visualization::stickTwoImageHorizontally( const cv::Mat& refImg, const cv::Mat& curImg, cv::Mat& img )
{
    cv::hconcat( refImg, curImg, img );
}