#ifndef __UTILS_HPP__
#define __UTILS_HPP__

#include "frame.hpp"
#include "feature.hpp"
#include "point.hpp"

#include <experimental/filesystem>
#include <iostream>
#include <string>

#include <Eigen/Core>
#include <opencv2/core.hpp>

namespace utils
{
Eigen::IOFormat eigenFormat();

Eigen::IOFormat eigenFormatIO();

std::string findAbsoluteFilePath( const std::string& relativeFilePath );

void listImageFilesInFolder( const std::string& imageFolderPath, std::vector< std::string >& imageListPath );

void cleanFolder( const std::string& imageFolderPath );

void writeAllInfoFile (const std::shared_ptr< Frame >& refFrame,
                    const std::shared_ptr< Frame >& curFrame,
                    std::ofstream& fileWriter);

void writeFeaturesInfoFile (const std::shared_ptr< Frame >& refFrame,
                    const std::shared_ptr< Frame >& curFrame,
                    std::ofstream& fileWriter);

void readAllFromFile (std::shared_ptr< Frame >& refFrame,
                    std::shared_ptr< Frame >& curFrame,
                    std::ifstream& fileReader);

void readFeaturesFromFile (std::shared_ptr< Frame >& refFrame,
                    std::shared_ptr< Frame >& curFrame,
                    std::ifstream& fileReader);

namespace constants
{
inline constexpr double pi{ 3.141592653589793 };       // inline constexpr is C++17 or newer only
inline constexpr double inv_pi{ 0.3183098861837907 };  // inline constexpr is C++17 or newer only
inline constexpr double pi_2{ 1.5707963267948966 };    // inline constexpr is C++17 or newer only
inline constexpr double pi_4{ 0.7853981633974483 };    // inline constexpr is C++17 or newer only
inline constexpr double sqrt_2_pi{ 2.5066282746310002 };
inline constexpr double inv_sqrt_2_pi{ 0.3989422804014327 };
}  // namespace constants

}  // namespace utils

#endif /* _UTILS_HPP__ */