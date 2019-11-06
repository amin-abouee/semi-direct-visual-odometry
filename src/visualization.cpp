#include "visualization.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

void Visualization::visualizeEpipole( const cv::Mat& img,
                                      const Eigen::Vector3d& vec,
                                      const Eigen::Matrix3d& K,
                                      const std::string& windowsName )
{
    cv::Mat imgBGR;
    if ( img.channels() == 1 )
        cv::cvtColor( img, imgBGR, cv::COLOR_GRAY2BGR );
    else
        imgBGR = img.clone();

    const Eigen::Vector3d projected = K * vec;
    cv::circle( imgBGR, cv::Point2i( projected( 0 ), projected( 1 ) ), 5.0, cv::Scalar( 0, 255, 165 ) );
    cv::imshow( windowsName, imgBGR );
}

void Visualization::visualizeEpipolarLine( const cv::Mat& img,
                                           const Eigen::Vector3d& vec,
                                           const Eigen::Matrix3d& E,
                                           const std::string& windowsName )
{
    cv::Mat imgBGR;
    if ( img.channels() == 1 )
        cv::cvtColor( img, imgBGR, cv::COLOR_GRAY2BGR );
    else
        imgBGR = img.clone();

    // std::cout << "Begin" << std::endl;
    // std::cout << "vec: " << vec << std::endl;

    Eigen::Vector3d line = E * vec;
    // std::cout << "line: " << line << std::endl;

    double nu = line( 0 ) * line( 0 ) + line( 1 ) * line( 1 );
    nu        = 1 / std::sqrt( nu );
    line *= nu;

    const cv::Point p1( 0, -line( 2 ) / line( 1 ) );
    // std::cout << "p1: " << p1 << std::endl;

    const cv::Point p2( imgBGR.cols - 1, -( line( 2 ) + line( 0 ) * ( imgBGR.cols - 1 ) ) / line( 1 ) );
    // std::cout << "p2: " << p2 << std::endl;

    cv::line( imgBGR, p1, p2, cv::Scalar( 0, 160, 200 ) );

    // std::cout << "Begin" << std::endl;
    // std::cout << "vec: " << vec << std::endl;
    // Eigen::Vector3d normVex = vec / vec.norm();
    // std::cout << "normVex: " << normVex << std::endl;
    // Eigen::Vector3d v1 = normVex * 0.2;
    // Eigen::Vector3d v2 = normVex * 0.5;
    // Eigen::Vector3d p1 = K * (R * v1 + t);
    // Eigen::Vector3d p2 = K * (R * v2 + t);
    // std::cout << "p1: " << p1 << std::endl;
    // std::cout << "p2: " << p2 << std::endl;
    // p1 /= p1[2];
    // p2 /= p2[2];
    // std::cout << "p1: " << p1 << std::endl;
    // std::cout << "p2: " << p2 << std::endl;
    // cv::line(imgBGR, cv::Point(p1(0), p1(1)), cv::Point(p2(0), p2(1)), cv::Scalar(0, 255, 165));
    cv::imshow( windowsName, imgBGR );
}

void Visualization::visualizeEpipolarLines( const cv::Mat& img,
                                           const Eigen::MatrixXd& vecs,
                                           const Eigen::Matrix3d& E,
                                           const std::string& windowsName )
{
    cv::Mat imgBGR;
    if ( img.channels() == 1 )
        cv::cvtColor( img, imgBGR, cv::COLOR_GRAY2BGR );
    else
        imgBGR = img.clone();

    for(std::size_t i(0); i< vecs.cols(); i++)
    {
        Eigen::Vector3d line = E * vecs.col(i);
        double nu = line( 0 ) * line( 0 ) + line( 1 ) * line( 1 );
        nu        = 1 / std::sqrt( nu );
        line *= nu;
        const cv::Point p1( 0, -line( 2 ) / line( 1 ) );
        const cv::Point p2( imgBGR.cols - 1, -( line( 2 ) + line( 0 ) * ( imgBGR.cols - 1 ) ) / line( 1 ) );
        cv::line( imgBGR, p1, p2, cv::Scalar( 0, 160, 200 ) );
    }
    cv::imshow( windowsName, imgBGR );
}