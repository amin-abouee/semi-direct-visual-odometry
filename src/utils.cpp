#include "utils.hpp"
#include <iomanip>
#include <fstream>

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
            imageListPath.emplace_back( file.path().string() );
        }
    }

    std::sort( imageListPath.begin(), imageListPath.end() );
}

void utils::cleanFolder( const std::string& imageFolderPath )
{
    std::filesystem::path path( findAbsoluteFilePath( imageFolderPath ) );
    std::cout << "path: " << path.string() << std::endl;
    auto numberDeleted = std::filesystem::remove_all( path );
    std::cout << "numberDeleted: " << numberDeleted << std::endl;
}

void utils::writeAllInfoFile( const std::shared_ptr< Frame >& refFrame, const std::shared_ptr< Frame >& curFrame, std::ofstream& fileWriter )
{
    for ( uint32_t i( 0 ); i < refFrame->numberObservation(); i++ )
    {
        const Eigen::Vector2d refFeature = refFrame->m_features[ i ]->m_pixelPosition;
        const Eigen::Vector2d curFeature = curFrame->m_features[ i ]->m_pixelPosition;
        const Eigen::Vector3d point                = refFrame->m_features[ i ]->m_point->m_position;
        fileWriter << std::setprecision( 6 ) << refFeature.x() << " " << refFeature.y() << " " << curFeature.x() << " " << curFeature.y()
                   << " " << point.x() << " " << point.y() << " " << point.z() << std::endl;
    }
}

void utils::writeFeaturesInfoFile( const std::shared_ptr< Frame >& refFrame, const std::shared_ptr< Frame >& curFrame, std::ofstream& fileWriter )
{
    for ( uint32_t i( 0 ); i < refFrame->numberObservation(); i++ )
    {
        const Eigen::Vector2d refFeature = refFrame->m_features[ i ]->m_pixelPosition;
        const Eigen::Vector2d curFeature = curFrame->m_features[ i ]->m_pixelPosition;
        fileWriter << std::setprecision( 6 ) << refFeature.x() << " " << refFeature.y() << " " << curFeature.x() << " " << curFeature.y() << std::endl;
    }
}

void utils::readAllFromFile( std::shared_ptr< Frame >& refFrame, std::shared_ptr< Frame >& curFrame, std::ifstream& fileReader )
{
    refFrame->m_features.clear();
    curFrame->m_features.clear();
    for ( uint32_t i( 0 ); i < refFrame->numberObservation(); i++ )
    {
        Eigen::Vector2d refFeaturePos;
        Eigen::Vector2d curFeaturePos;
        Eigen::Vector3d pointPose;
        fileReader >> refFeaturePos.x() >> refFeaturePos.y() >> curFeaturePos.x() >> curFeaturePos.y() >> pointPose.x() >> pointPose.y() >>
          pointPose.z();
        std::shared_ptr< Feature > refFeature = std::make_shared< Feature >( refFrame, refFeaturePos, 0.0, 0.0, 0 );
        std::shared_ptr< Feature > curfeature = std::make_shared< Feature >( curFrame, curFeaturePos, 0.0, 0.0, 0 );
        std::shared_ptr< Point > point        = std::make_shared< Point >( pointPose );

        refFrame->addFeature( refFeature );
        curFrame->addFeature( curfeature );

        point->addFeature( refFeature );
        point->addFeature( curfeature );

        refFeature->setPoint( point );
        curfeature->setPoint( point );
    }
}

void utils::readFeaturesFromFile( std::shared_ptr< Frame >& refFrame, std::shared_ptr< Frame >& curFrame, std::ifstream& fileReader )
{
    refFrame->m_features.clear();
    curFrame->m_features.clear();
    for ( uint32_t i( 0 ); i < refFrame->numberObservation(); i++ )
    {
        Eigen::Vector2d refFeaturePos;
        Eigen::Vector2d curFeaturePos;
        fileReader >> refFeaturePos.x() >> refFeaturePos.y() >> curFeaturePos.x() >> curFeaturePos.y();
        std::shared_ptr< Feature > refFeature = std::make_shared< Feature >( refFrame, refFeaturePos, 0.0, 0.0, 0 );
        std::shared_ptr< Feature > curfeature = std::make_shared< Feature >( curFrame, curFeaturePos, 0.0, 0.0, 0 );

        refFrame->addFeature( refFeature );
        curFrame->addFeature( curfeature );
    }
}
