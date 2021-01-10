/**
 * Project Inside Out Tracking
 * @version 0.6.0
 *
 * @file image_pyramid.h
 * @brief class for creating the image pyramid
 *
 * @date 22.02.2018
 * @author Amin Abouee
 *
 * @section DESCRIPTION
 *
 *
 */

#ifndef _IMAGEPYRAMID_H
#define _IMAGEPYRAMID_H

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

/// @brief Image pyramid is a stack of images with different resolutions
class ImagePyramid final
{
public:
    /// @brief Construct a new image pyramid object
    ///
    /// @param[in] maxPyramidLevel Maximum level of image pyramid. The image pyramid defines between level 0 (finest) and level
    /// maxImagePyramid (coarsest)
    explicit ImagePyramid( const std::size_t maxPyramidLevel );

    /// @brief Construct a new image pyramid object
    ///
    /// @param[in] baseImage image data at level 0 (original resolution)
    /// @param[in] maxPyramidLevel Maximum level of image pyramid. The image pyramid defines between level 0 (finest) and level
    /// maxImagePyramid (coarsest)
    explicit ImagePyramid( const cv::Mat& baseImage, const std::size_t maxPyramidLevel );

    /// @brief Copy construct a new image pyramid object
    ///
    /// @param[in] rhs
    ImagePyramid( const ImagePyramid& rhs ) = delete;

    /// @brief Move construct a new image pyramid object
    ///
    /// @param[in] rhs
    ImagePyramid( ImagePyramid&& rhs ) = delete;

    /// @brief Copy assignment operator
    ///
    /// @param[in] rhs
    /// @return ImagePyramid&
    ImagePyramid& operator=( const ImagePyramid& rhs ) = delete;

    /// @brief Move assignment operator
    ///
    /// @param[in] rhs
    /// @return ImagePyramid&
    ImagePyramid& operator=( ImagePyramid&& rhs ) = delete;

    // D'tor
    ~ImagePyramid() = default;

    /// @brief Create an image pyramid object
    ///
    /// @param[in] baseImage image data at level 0 (original resolution)
    /// @param[in] maxPyramidLevel Maximum level of image pyramid. The image pyramid defines between level 0 (finest) and level
    /// maxImagePyramid (coarsest)
    void createImagePyramid( const cv::Mat& baseImage, const std::size_t maxPyramidLevel );

    /// @brief Get the const reference of all images (all levels)
    ///
    /// @return const std::vector<cv::Mat>& Const reference of images in all levels 0 -> maxPyramidLevel
    const std::vector< cv::Mat >& getAllImages() const;

    /// @brief Get the reference of all images
    ///
    /// @return std::vector<cv::Mat>& Reference of images in all levels 0 -> maxPyramidLevel
    std::vector< cv::Mat >& getAllImages();

    /// @brief Get the const reference of image data at required level
    ///
    /// @param[in] level required level
    /// @return const cv::Mat& Const reference to required image level
    const cv::Mat& getImageAtLevel( const std::size_t level ) const;

    /// @brief Get the reference of image data at required level
    ///
    /// @param[in] level Required level
    /// @return cv::Mat& Reference to required image level
    cv::Mat& getImageAtLevel( const std::size_t level );

    /// @brief Get the const reference image at level 0 (base or original image)
    ///
    /// @return const cv::Mat& Const reference to image at level 0
    const cv::Mat& getBaseImage() const;

    /// @brief Get the reference image at level 0 (base or original image)
    ///
    /// @return cv::Mat& Reference to image at level 0
    cv::Mat& getBaseImage();

	/// @brief Get the const reference gradient image at level 0 (base or original gradient image)
    ///
    /// @return const cv::Mat& Const reference to gradient image at level 0
    const cv::Mat& getBaseGradientImage() const;

    /// @brief Get the reference gradient image at level 0 (base or original gradient image)
    ///
    /// @return cv::Mat& Reference to gradient image at level 0
    cv::Mat& getBaseGradientImage();

       /// @brief Get the const reference of gradient data at required level
    ///
    /// @param[in] level required level
    /// @return const cv::Mat& Const reference to required gradient level
    const cv::Mat& getGradientAtLevel( const std::size_t level ) const;

    /// @brief Get the reference of gradient data at required level
    ///
    /// @param[in] level Required level
    /// @return cv::Mat& Reference to required gradient level
    cv::Mat& getGradientAtLevel( const std::size_t level );

    /// @brief Get the size of image pyramid (maxPyramidLevel + 1)
    ///
    /// @return std::size_t Size of image pyramid
    std::size_t getSizeImagePyramid() const;

    /// @brief Get the image resolution of image at required level
    ///
    /// @param[in] level Required level
    /// @return cv::Size Image resolution
    cv::Size getImageSizeAtLevel( const std::size_t level );

    /// @brief Get the image resolution of image at level 0 (base image)
    ///
    /// @return cv::Size Image resolution of base image
    cv::Size getBaseImageSize() const;

    /// @brief Clear the images
    void clear();

private:
    std::size_t m_baseImageWidth  = 0;           ///< Width of image at level 0 (base image)
    std::size_t m_baseImageHeight = 0;           ///< Height of image at level 0 (base image)
    std::vector< cv::Mat > m_vecImages;          ///< Stack of images with different resolutions from level 0 -> maxPyramidLevel
    std::vector< cv::Mat > m_vecGradientImages;  ///< Stack of gradient images with different resolutions from level 0 -> maxPyramidLevel
};

#endif  //_IMAGEPYRAMID_H
