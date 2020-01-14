#ifndef __UTILS_HPP__
#define __UTILS_HPP__

#include <iostream>
#include <string>
#include <experimental/filesystem>

#include <Eigen/Core>
#include <opencv2/core.hpp>

namespace utils
{
    namespace fs = std::experimental::filesystem;

    Eigen::IOFormat eigenFormat();

    std::string findAbsoluteFilePath(const std::string& relativeFilePath);
}

#endif /* _UTILS_HPP__ */