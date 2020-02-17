#include "config.hpp"
#include <fstream>


// std::unique_ptr< Config > Config::m_instance;
// std::once_flag Config::m_onceFlag;

// Config* Config::getInstance()
// {
//     return m_instance.get();
// }

// Config* Config::init( const std::string& configFile )
// {
//     std::call_once( m_onceFlag, [&]() { m_instance = std::make_unique< Config >( configFile ); } );
//     return m_instance.get();
// }

Config::Config( const std::string& configFile )
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

    const nlohmann::json& filePathJson = m_configJson[ "file_paths" ];
    m_logFilePath = filePathJson[ "log_file" ].get<std::string>();

    const nlohmann::json& cameraJson = m_configJson[ "camera" ];
    m_cameraCalibrationPath = cameraJson[ "camera_calibration" ].get< std::string >();
    m_imgWidth              = cameraJson[ "img_width" ].get< int32_t >();
    m_imgHeight             = cameraJson[ "img_height" ].get< int32_t >();

    const nlohmann::json& algoJson = m_configJson[ "algorithm" ];
    m_gridPixelSize                = algoJson[ "grid_size_select_features" ].get< uint32_t >();
    m_patchSizeOpticalFlow         = algoJson[ "patch_size_optical_flow" ].get< uint32_t >();
    m_patchSizeImageAlignment      = algoJson[ "patch_size_image_alignment" ].get< uint32_t >();
    m_minLevelImagePyramid         = algoJson[ "min_level_image_pyramid" ].get< uint32_t >();
    m_maxLevelImagePyramid         = algoJson[ "max_level_image_pyramid" ].get< uint32_t >();
}