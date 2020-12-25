#ifndef CONFIG_HPP
#define CONFIG_HPP

#include <nlohmann/json.hpp>

#include <iostream>
#include <memory>
#include <string>

// https://stackoverflow.com/a/55296858/1804533
// https://stackoverflow.com/a/34653512/1804533
// http://www.nuonsoft.com/blog/2012/10/21/implementing-a-thread-safe-singleton-with-c11/
// https://stackoverflow.com/a/52308483/1804533

/// @brief VO Algorithm configuration class.
class Config final
{
public:
    /// @brief Construct a new Config object
    ///
    /// @param[in] configFile
    explicit Config( const std::string& configFile );

    // Copy C'tor
    Config( const Config& rhs ) = delete;  // non construction-copyable

    // Move C'tor
    Config( Config&& rhs ) = delete;  // non construction movable

    // Copy assignment operator
    Config& operator=( const Config& rhs ) = delete;  // non copyable

    // Move assignment operator
    Config& operator=( Config&& rhs ) = delete;  // non movable

    // D'tor
    ~Config() = default;

    nlohmann::json m_configJson;  ///< Json node to the head

    std::string m_logFilePath;            ///< Path of easylogging log file
    std::string m_cameraCalibrationPath;  ///< Path of camera calibration file

    int32_t m_imgWidth;   ///< Image width in pixel
    int32_t m_imgHeight;  ///< Image Height in pixel

    bool m_enableVisualization;  ///< Enable visualization
    std::string m_savingType;    ///< Type of visualization type, File, LiveShow

    int32_t m_patchSizeOpticalFlow;                     ///< Patch size of optical flow
    int32_t m_thresholdGradientMagnitude;               ///< Minimum gradient magnitude threshold to select points
    uint32_t m_minDetectedPointsSuccessInitialization;  ///< Minimum number points for success initialization
    uint32_t m_desiredDetectedPointsForInitialization;  ///< Size of Desired detected points in initialization
    double m_initMapScaleFactor;                        ///< Scale factor of initialize map
    int32_t m_disparityThreshold;                       ///< initialize minimum disparity threshold

    int32_t m_cellPixelSize;            ///< Cell size in pixel for grid
    int32_t m_patchSizeImageAlignment;  ///< Patch size in pixel for image alignment
    int32_t m_minLevelImagePyramid;     ///< Minimim level of image pyramid (0 = finest)
    int32_t m_maxLevelImagePyramid;     ///< Maximim level of image pyramid (n = coarsest)

private:
};

#endif /* CONFIG_HPP */
