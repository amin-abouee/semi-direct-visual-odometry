#ifndef __UTILS_HPP__
#define __UTILS_HPP__

#include <experimental/filesystem>
#include <iostream>
#include <string>

#include <Eigen/Core>
#include <opencv2/core.hpp>

namespace utils
{
namespace fs = std::experimental::filesystem;

Eigen::IOFormat eigenFormat();

std::string findAbsoluteFilePath( const std::string& relativeFilePath );
}  // namespace utils

#endif /* _UTILS_HPP__ */