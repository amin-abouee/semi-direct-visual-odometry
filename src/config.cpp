#include "config.hpp"
#include <fstream>

Config::Config(const std::string& configFile)
{
    nlohmann::json configParser;
    try
    {
        std::ifstream fileReader( configFile );
        m_configJson = nlohmann::json::parse( fileReader );
        fileReader.close();
    }
    catch ( const std::exception& e )
    {
        std::cerr << "JSON Parser Error:" << e.what() << '\n';
    }

    const nlohmann::json& cameraJson  = m_configJson[ "camera" ];
    std::string m_logFilePath;
    std::string m_cameraCalibrationPath = cameraJson[ "camera_calibration" ].get< std::string >();
    uint32_t m_imgWidth = cameraJson[ "img_width" ].get< int32_t >();
    uint32_t m_imgHeight = cameraJson[ "img_height" ].get< int32_t >();

    const nlohmann::json& algoJson = m_configJson[ "algorithm" ];
    uint32_t m_gridPixelSize = algoJson[ "grid_size_select_features" ].get< uint32_t >();
    uint32_t m_patchSizeOpticalFlow = algoJson[ "patch_size_optical_flow" ].get< uint32_t >();
    uint32_t m_patchSizeImageAlignment = algoJson[ "patch_size_image_alignment" ].get< uint32_t >();
    uint32_t m_minLevelImagePyramid = algoJson[ "min_level_image_pyramid" ].get< uint32_t >();
    uint32_t m_maxLevelImagePyramid = algoJson[ "max_level_image_pyramid" ].get< uint32_t >();
}