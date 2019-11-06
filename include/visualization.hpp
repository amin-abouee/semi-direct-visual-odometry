/**
 * @file visualization.hpp
 * @brief visualization class for each cases
 *
 * @date 06.11.2019
 * @author Amin Abouee
 *
 * @section DESCRIPTION
 *
 *
 */
#ifndef __VISUALIZATION_H__
#define __VISUALIZATION_H__

#include <iostream>
#include <string>

#include <Eigen/Core>

#include <opencv2/core.hpp>

class Visualization final
{
public:
    // C'tor
    explicit Visualization() = default;
    // Copy C'tor
    Visualization( const Visualization& rhs ) = default;
    // move C'tor
    Visualization( Visualization&& rhs ) = default;
    // Copy assignment operator
    Visualization& operator=( const Visualization& rhs ) = default;
    // move assignment operator
    Visualization& operator=( Visualization&& rhs ) = default;
    // D'tor
    ~Visualization() = default;

    void visualizeEpipole( const cv::Mat& img,
                           const Eigen::Vector3d& vec,
                           const Eigen::Matrix3d& K,
                           const std::string& windowsName );

    void visualizeEpipolarLine( const cv::Mat& img,
                                const Eigen::Vector3d& vec,
                                const Eigen::Matrix3d& E,
                                const std::string& windowsName );

    void visualizeEpipolarLines( const cv::Mat& img,
                                 const Eigen::MatrixXd& vecs,
                                 const Eigen::Matrix3d& E,
                                 const std::string& windowsName );

private:
};

#endif /* __VISUALIZATION_H__ */