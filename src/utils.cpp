#include "utils.hpp"

Eigen::IOFormat utils::eigenFormat()
{
    return Eigen::IOFormat( 6, Eigen::DontAlignCols, ", ", " , ", "[", "]", "[", "]" );
}

std::string utils::findAbsoluteFilePath( const std::string& relativeFilePath )
{
    // https://stackoverflow.com/a/56645283/1804533
    fs::path finalPath;
    auto currentPath = fs::current_path();

    // https://stackoverflow.com/a/24735243/1804533
    for ( const auto& part : fs::path( currentPath ) )
    {
        if ( part.string() == "build" || part.string() == "bin" )
            break;
        else
            finalPath /= part;
    }

    finalPath /= relativeFilePath;
    return finalPath.string();
}