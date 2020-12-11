#ifndef __CONFIG_HPP__
#define __CONFIG_HPP__

#include <iostream>
#include <string>
#include <memory>

#include <nlohmann/json.hpp>


// https://stackoverflow.com/a/55296858/1804533
// https://stackoverflow.com/a/34653512/1804533
// http://www.nuonsoft.com/blog/2012/10/21/implementing-a-thread-safe-singleton-with-c11/
// https://stackoverflow.com/a/52308483/1804533
class Config final
{
public:
    explicit Config(const std::string& configFile);
    Config (const Config& rhs) = delete;
    Config (Config&& rhs) = delete;
    Config& operator= (const Config& rhs) = delete;
    Config& operator= (Config&& rhs) = delete;

    nlohmann::json m_configJson;
    std::string m_logFilePath;
    std::string m_cameraCalibrationPath;
    uint32_t m_imgWidth;
    uint32_t m_imgHeight;
    uint32_t m_cellPixelSize;
    uint32_t m_patchSizeOpticalFlow;
    uint32_t m_patchSizeImageAlignment;
    uint32_t m_minLevelImagePyramid;
    uint32_t m_maxLevelImagePyramid;

private:
};

#endif /* __CONFIG_HPP__ */

