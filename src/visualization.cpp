#include "visualization.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

void Visualization::visualizeFeaturePoints( const cv::Mat& img,
                                            const Eigen::MatrixXd& vecs,
                                            const std::string& windowsName )
{
    cv::Mat normMag, imgBGR;
    cv::normalize( img, normMag, 0, 255, cv::NORM_MINMAX, CV_8UC1 );
    cv::cvtColor( normMag, imgBGR, cv::COLOR_GRAY2BGR );

    for ( int i( 0 ); i < vecs.cols(); i++ )
    {
        cv::circle( imgBGR, cv::Point2i( vecs.col( i )( 0 ), vecs.col( i )( 1 ) ), 2.0, cv::Scalar( 0, 255, 0 ) );
    }
    cv::imshow( windowsName, imgBGR );
}

void Visualization::visualizeGrayImage(const cv::Mat& img, const std::string& windowsName)
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

void Visualization::visualizeHSVColoredImage(const cv::Mat& img, const std::string& windowsName)
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
                cv::Vec3b& px = imgHSV.at< cv::Vec3b >( cv::Point( j, i ) );
                cv::Scalar color =
                  generateColor( minMagnitude, 255.0, img.at< float >( i, j ) - minMagnitude );
                px[ 0 ] = color[ 0 ];
                px[ 1 ] = color[ 1 ];
                px[ 2 ] = color[ 2 ];
            }
        }
    }
    cv::Mat imgHSVNew;
    cv::cvtColor( imgHSV, imgHSVNew, cv::COLOR_HSV2BGR );
    cv::imshow( windowsName, imgHSVNew );
}

void Visualization::visualizeEpipole( const cv::Mat& img,
                                      const Eigen::Vector3d& vec,
                                      const Eigen::Matrix3d& K,
                                      const std::string& windowsName )
{
    // https://answers.opencv.org/question/182587/how-to-draw-epipolar-line/
    cv::Mat imgBGR = getBGRImage(img);

    const Eigen::Vector3d projected = K * vec;
    cv::circle( imgBGR, cv::Point2i( projected( 0 ), projected( 1 ) ), 5.0, cv::Scalar( 0, 255, 165 ) );
    cv::imshow( windowsName, imgBGR );
}

void Visualization::visualizeEpipolarLine( const cv::Mat& img,
                                           const Eigen::Vector3d& vec,
                                           const Eigen::Matrix3d& F,
                                           const std::string& windowsName )
{
    cv::Mat imgBGR = getBGRImage(img);

    Eigen::Vector3d line = F * vec;

    double nu = line( 0 ) * line( 0 ) + line( 1 ) * line( 1 );
    nu        = 1 / std::sqrt( nu );
    line *= nu;

    const cv::Point p1( 0, -line( 2 ) / line( 1 ) );
    const cv::Point p2( imgBGR.cols - 1, -( line( 2 ) + line( 0 ) * ( imgBGR.cols - 1 ) ) / line( 1 ) );
    cv::line( imgBGR, p1, p2, cv::Scalar( 0, 160, 200 ) );
    cv::imshow( windowsName, imgBGR );
}

void Visualization::visualizeEpipolarLines( const cv::Mat& img,
                                            const Eigen::MatrixXd& vecs,
                                            const Eigen::Matrix3d& F,
                                            const std::string& windowsName )
{
    cv::Mat imgBGR = getBGRImage(img);

    for ( std::size_t i( 0 ); i < vecs.cols(); i++ )
    {
        Eigen::Vector3d line = F * vecs.col( i );
        double nu            = line( 0 ) * line( 0 ) + line( 1 ) * line( 1 );
        nu                   = 1 / std::sqrt( nu );
        line *= nu;
        const cv::Point p1( 0, -line( 2 ) / line( 1 ) );
        const cv::Point p2( imgBGR.cols - 1, -( line( 2 ) + line( 0 ) * ( imgBGR.cols - 1 ) ) / line( 1 ) );
        cv::line( imgBGR, p1, p2, cv::Scalar( 0, 160, 200 ) );
    }
    cv::imshow( windowsName, imgBGR );
}

cv::Scalar Visualization::generateColor( const double min, const double max, const float value )
{
    int hue = ( 120 / ( max - min ) ) * value;
    return cv::Scalar( hue, 100, 100 );
}

cv::Mat Visualization::getBGRImage(const cv::Mat& img)
{
    cv::Mat imgBGR;
    if ( img.channels() == 1 )
        cv::cvtColor( img, imgBGR, cv::COLOR_GRAY2BGR );
    else
        imgBGR = img.clone();
    return imgBGR;
}