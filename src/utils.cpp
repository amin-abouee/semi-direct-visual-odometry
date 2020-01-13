#include "utils.hpp"

Eigen::IOFormat utils::eigenFormat()
{
    return Eigen::IOFormat( 6, Eigen::DontAlignCols, ", ", " , ", "[", "]", "[", "]" );
}

std::string utils::findAbsoluteFilePath(std::string& relativeFilePath)
{
    // https://stackoverflow.com/a/56645283/1804533

    std::cout << std::experimental::filesystem::current_path().string() << std::endl;
    // if (fs::exists(relativeFilePath))
    return relativeFilePath;
}