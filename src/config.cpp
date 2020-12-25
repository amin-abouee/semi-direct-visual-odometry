#include "config.hpp"
#include "utils.hpp"

#include <easylogging++.h>

#include <fstream>

#define CONFIG_LOG( LEVEL ) CLOG( LEVEL, "Config" )
#define CONFIG_LOG_IF( condition, LEVEL ) CLOG_IF( condition, LEVEL, "Config" )

Config::Config( const std::string& configFile )
{
    el::Configurations defaultConf;
    defaultConf.setGlobally( el::ConfigurationType::Format,
                             "[%datetime{%d.%M.%Y %H:%m:%s}] [%logger:%level:%thread] [%fbase:%line]: %msg" );
    el::Loggers::reconfigureLogger( "Config", defaultConf );
    el::Loggers::addFlag( el::LoggingFlag::ColoredTerminalOutput );

    nlohmann::json configParser;
    try
    {
        std::ifstream fileReader( configFile );
        m_configJson = nlohmann::json::parse( fileReader );
        fileReader.close();
    }
    catch ( const std::exception& e )
    {
        CONFIG_LOG( FATAL ) << "JSON Parser Error:" << e.what();
    }

    // read file paths params
    const nlohmann::json& filePathJson = m_configJson[ "file_paths" ];

    m_cameraCalibrationPath = utils::findAbsoluteFilePath( filePathJson[ "camera_calibration_file" ].get< std::string >() );
    CONFIG_LOG_IF( m_cameraCalibrationPath.empty(), FATAL ) << "Camera calibration path is empty. Please update your params.json";

    m_logFilePath = utils::findAbsoluteFilePath( filePathJson[ "log_file" ].get< std::string >() );
    CONFIG_LOG_IF( m_logFilePath.empty(), FATAL ) << "log config path is empty. Please update your params.json";

    // read camera params
    const nlohmann::json& cameraJson = m_configJson[ "camera" ];

    m_imgWidth = cameraJson[ "img_width" ].get< int32_t >();
    CONFIG_LOG_IF( m_imgWidth <= 0, FATAL ) << "image width is <= 0. Please update your camera calibration json file";

    m_imgHeight = cameraJson[ "img_height" ].get< int32_t >();
    CONFIG_LOG_IF( m_imgHeight <= 0, FATAL ) << "image height is <= 0. Please update your camera calibration json file";

    // read visualization params
    const nlohmann::json& visJson = m_configJson[ "visualization" ];

    m_enableVisualization         = visJson[ "enable_visualization" ].get< bool >();
    m_savingType                  = visJson[ "saving_type" ].get< std::string >();

    // read initialization params
    const nlohmann::json& initJson = m_configJson[ "initialization" ];
    m_patchSizeOpticalFlow         = initJson[ "patch_size_optical_flow" ].get< int32_t >();
    CONFIG_LOG_IF( m_patchSizeOpticalFlow <= 0, FATAL ) << "Patch size for optical flow is <= 0. Please update your params.json";

    m_thresholdGradientMagnitude = initJson[ "threshold_gradient_magnitude" ].get< int32_t >();
    CONFIG_LOG_IF( m_thresholdGradientMagnitude <= 0 || m_thresholdGradientMagnitude > 255, FATAL )
      << "Gradient magnitude threshold have to set between [0-255]. Please update your params.json";

    m_minDetectedPointsSuccessInitialization = initJson[ "min_detected_points" ].get< uint32_t >();
    CONFIG_LOG_IF( m_minDetectedPointsSuccessInitialization <= 0, FATAL )
      << "Minimum number points for success initialization is <= 0. Please update your params.json";

    m_desiredDetectedPointsForInitialization = initJson[ "desired_detected_points" ].get< uint32_t >();
    CONFIG_LOG_IF( m_desiredDetectedPointsForInitialization <= 0, FATAL )
      << "Size of Desired detected points in initialization is <= 0. Please update your params.json";

    m_initMapScaleFactor = initJson[ "map_scale_factor" ].get< double >();
    CONFIG_LOG_IF( m_initMapScaleFactor <= 0.0, FATAL ) << "Map scale factor for initialization is <= 0. Please update your params.json";

    m_disparityThreshold = initJson[ "disparity_threshold" ].get< double >();
    CONFIG_LOG_IF( m_disparityThreshold <= 0.0, FATAL ) << "Disparity threshold is <= 0. Please update your params.json";

    // read algorithm params
    const nlohmann::json& algoJson = m_configJson[ "algorithm" ];
    m_cellPixelSize                = algoJson[ "cell_pixel_size" ].get< int32_t >();
    CONFIG_LOG_IF( m_cellPixelSize <= 0, FATAL ) << "Cell pixel size is <= 0. Please update your params.json";

    m_patchSizeImageAlignment = algoJson[ "patch_size_image_alignment" ].get< int32_t >();
    CONFIG_LOG_IF( m_patchSizeImageAlignment <= 0, FATAL ) << "Patch size image alignment is <= 0. Please update your params.json";

    m_minLevelImagePyramid = algoJson[ "min_level_image_pyramid" ].get< int32_t >();
    CONFIG_LOG_IF( m_minLevelImagePyramid < 0, FATAL ) << "Minimum level of image pyramid is < 0. Please update your params.json";

    m_maxLevelImagePyramid = algoJson[ "max_level_image_pyramid" ].get< int32_t >();
    CONFIG_LOG_IF( m_maxLevelImagePyramid < 0, FATAL ) << "Maximum level of image pyramid is < 0. Please update your params.json";
}