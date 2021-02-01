#include "utils.hpp"

Eigen::IOFormat utils::eigenFormat()
{
    return Eigen::IOFormat( 6, Eigen::DontAlignCols, ", ", " , ", "[", "]", "[", "]" );
}

Eigen::IOFormat utils::eigenFormatIO()
{
    return Eigen::IOFormat( 6, Eigen::DontAlignCols, " ", " ", "", "", "", "" );
}

std::string utils::findAbsoluteFilePath( const std::string& relativeFilePath )
{
    // https://stackoverflow.com/a/56645283/1804533
    std::filesystem::path finalPath;
    auto currentPath = std::filesystem::current_path();
        // https://stackoverflow.com/a/24735243/1804533
    for ( const auto& part : std::filesystem::path( currentPath ) )
    {
        if ( part.string() == "build" || part.string() == "bin" )
            break;
        else
            finalPath /= part;
    }
    

    finalPath /= relativeFilePath;
    return finalPath.string();
}

void utils::listImageFilesInFolder( const std::string& imageFolderPath, std::vector< std::string >& imageListPath )
{
    for ( const auto& file : std::filesystem::directory_iterator( imageFolderPath ) )
    {
        if ( file.path().extension() == ".png" || file.path().extension() == "jpg" )
        {
            imageListPath.emplace_back(file.path().string());
        }
    }

    std::sort(imageListPath.begin(), imageListPath.end());
}

void utils::cleanFolder( const std::string& imageFolderPath )
{
    std::filesystem::path path(findAbsoluteFilePath(imageFolderPath));
    std::cout << "path: " << path.string() << std::endl;
    auto numberDeleted = std::filesystem::remove_all(path);
    std::cout << "numberDeleted: " << numberDeleted << std::endl;
}
