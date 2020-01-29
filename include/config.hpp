#ifndef __CONFIG_HPP__
#define __CONFIG_HPP__

#include <iostream>
#include <string>

// https://stackoverflow.com/a/55296858/1804533
// https://stackoverflow.com/a/34653512/1804533

// https://stackoverflow.com/a/52308483/1804533
class Config final
{
public:
    Config (const Config& rhs) = delete;
    Config (Config&& rhs) = delete;
    Config& operator= (const Config& rhs) = delete;
    Config& operator= (Config&& rhs) = delete;
    ~Config() = default;

    static Config* getInstance()
    {
        std::call_once(m_once, []() {
            m_instance.reset(new T);
        });
        return m_instance.get();
    }

    static Config* getInstanceString(const std::string& configFile)
    {
        std::call_once(m_once, [&]() {
            m_instance.reset(new Config(configFile));
        });
        return m_instance.get();
    }

private:
    explicit Config(const std::string& configFile);
    static std::unique_ptr<Config> m_instance;
    static std::once_flag m_once;

    std::string m_logFilePath;
    std::string m_cameraCalibrationPath;
    uint32_t m_imgWidth;
    uint32_t m_imgHeight;
    uint32_t m_gridPixelSize;
    uint32_t m_patchSizeOpticalFlow;
    uint32_t m_patchSizeImageAlignment;
    uint32_t m_minLevelImagePyramid;
    uint32_t m_maxLevelImagePyramid;

};

#endif /* __CONFIG_HPP__ */

