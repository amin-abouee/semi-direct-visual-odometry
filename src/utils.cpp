#include "utils.hpp"

Eigen::IOFormat utils::eigenFormat()
{
    return Eigen::IOFormat( 6, Eigen::DontAlignCols, ", ", " , ", "[", "]", "[", "]" );
}

std::string utils::findAbsoluteFilePath(std::string& relativeFilePath)
{
    if (fs::exists(relativeFilePath))
        return relativeFilePath;
}