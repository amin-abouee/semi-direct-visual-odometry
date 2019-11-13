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

class ImagePyramid final
{
public:
	// C'tor
	explicit ImagePyramid(const std::size_t level);
    // C'tor
	explicit ImagePyramid(const cv::Mat& baseImage, const std::size_t level );
	// Copy C'tor
	ImagePyramid( const ImagePyramid& rhs ) = default;
	// move C'tor
	ImagePyramid( ImagePyramid&& rhs ) = default;
	// Copy assignment operator
	ImagePyramid& operator=( const ImagePyramid& rhs ) = default;
	// move assignment operator
	ImagePyramid& operator=( ImagePyramid&& rhs ) = default;
	// D'tor
	~ImagePyramid() = default;

	/**
	 * [createImagePyramid description]
	 * @param baseImage [description]
	 * @param level     [description]
	 */
    void createImagePyramid(const cv::Mat& baseImage, const std::size_t level);

	/**
	 * [getAllImages description]
	 */
    const std::vector<cv::Mat>& getAllImages() const;
    std::vector<cv::Mat>& getAllImages();

	/**
	 * [getImageAtLevel description]
	 * @param  level [description]
	 * @return       [description]
	 */
    const cv::Mat& getImageAtLevel(const std::size_t level) const;
    cv::Mat& getImageAtLevel(const std::size_t level);

	/**
	 * [getBaseImage description]
	 * @return [description]
	 */
    const cv::Mat& getBaseImage() const;
    cv::Mat& getBaseImage();

	/**
	 * [getSizeImagePyramid description]
	 * @return [description]
	 */
    std::size_t getSizeImagePyramid() const;

	/**
	 * [getImageSizeAtLevel description]
	 * @param  level [description]
	 * @return       [description]
	 */
    cv::Size getImageSizeAtLevel(const std::size_t level);

	/**
	 * [getBaseImageSize description]
	 * @return [description]
	 */
    cv::Size getBaseImageSize() const;

	void clear();

private:
	std::size_t m_baseImageWidth  = 0;
	std::size_t m_baseImageHeight = 0;

	std::vector< cv::Mat > m_vecImages;
};

#endif  //_IMAGEPYRAMID_H
