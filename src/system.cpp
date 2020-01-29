#include "system.hpp"
#include "utils.hpp"

#include <opencv2/calib3d.hpp>

System::System(const nlohmann::json& jsonConfig)
{
    const nlohmann::json& cameraJson  = jsonConfig[ "camera" ];
    const std::string calibrationFile = utils::findAbsoluteFilePath( cameraJson[ "camera_calibration" ].get< std::string >() );
    cv::Mat cameraMatrix;
    cv::Mat distortionCoeffs;
    bool result = loadCameraIntrinsics( calibrationFile, cameraMatrix, distortionCoeffs );
    if ( result == false )
    {
        std::cout << "Failed to open the calibration file, check config.json file" << std::endl;
        // return EXIT_FAILURE;
    }

    const int32_t imgWidth  = cameraJson[ "img_width" ].get< int32_t >();
    const int32_t imgHeight = cameraJson[ "img_height" ].get< int32_t >();
}

bool System::loadCameraIntrinsics( const std::string& filename, cv::Mat& cameraMatrix, cv::Mat& distortionCoeffs )
{
    try
    {
        cv::FileStorage fs( filename, cv::FileStorage::READ );
        if ( !fs.isOpened() )
        {
            std::cout << "Failed to open " << filename << std::endl;
            return false;
        }

        fs[ "K" ] >> cameraMatrix;
        fs[ "d" ] >> distortionCoeffs;
        fs.release();
        return true;
    }
    catch ( std::exception& e )
    {
        std::cout << e.what() << std::endl;
        return false;
    }
}