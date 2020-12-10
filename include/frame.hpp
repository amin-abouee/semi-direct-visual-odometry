/**
 * @file Frame.hpp
 * @brief frame information
 *
 * @date 18.11.2019
 * @author Amin Abouee
 *
 * @section DESCRIPTION
 *
 *
 */
#ifndef __FRAME_HPP__
#define __FRAME_HPP__

#include <iostream>
#include <memory>

#include <Eigen/Core>
#include <sophus/se3.hpp>

#include "image_pyramid.hpp"
#include "pinhole_camera.hpp"

class Feature;

/**
 * @brief This class contains the RGB image informations, the absolute pose and its covariance and the camera geometry data
 * 
 *                                                                             
 *              Z                                                              
                ^                                                              
                |                                                              
                | World                                                        
                |                                                              
                O------>X                                                      
               / ^                                            ^                
              /   \                                          /                 
             v    |                                         /                  
            Y      \                                       /                   
                    \                                     /                    
                     \                                   /                     
                     |                   u              /                      
                      \               +----->---------------------+            
                       \              | Image                     |            
                       |            v | Pixel                     |            
                        \             v             x (cx, cy)    |            
                       t \            |            /              |            
                          \           |           /               |            
                          |           |          /                |            
                           \          +---------/-----------------+            
    Trans W to C = [R|t]    \                  /                               
                             \                /                                
                             |               /                                 
                              \             /                                  
                               \           /                                   
                               |          /                                    
                                \        /                                     
                                 \    Z /                                      
                                  \    /                                       
                                  |   /                                        
                                   \ /                                         
                                    O-------->X                                
                                    |                                          
                                    | Camera                                   
                                    |                                          
                                    v                                          
                                    Y                                          
 *
 */
class Frame final
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    static uint64_t m_frameCounter;
    uint64_t m_id;
    std::shared_ptr< PinholeCamera > m_camera;
    Sophus::SE3d m_absPose;
    Eigen::Matrix< double, 6, 6 > m_covPose;
    ImagePyramid m_imagePyramid;
    std::vector< std::shared_ptr< Feature > > m_features;
    bool m_keyFrame;
    double m_timestamp;

    // C'tor
    explicit Frame( const std::shared_ptr< PinholeCamera >& camera, const cv::Mat& img, const uint32_t maxImagePyramid, const double timestamp );
    // Copy C'tor
    Frame( const Frame& rhs ) = delete;  // non construction-copyable
    // move C'tor
    Frame( Frame&& rhs ) = delete;  // non construction movable
    // Copy assignment operator
    Frame& operator=( const Frame& rhs ) = delete;  // non copyable
    // move assignment operator
    Frame& operator=( Frame&& rhs ) = delete;  // movable
    // D'tor
    ~Frame() = default;

    /// Initialize new frame and create the image pyramid.
    void initFrame( const cv::Mat& img, const uint32_t maxImagePyramid );

    /// Select this frame as keyframe.
    void setKeyframe();

    /// Add a feature to the image
    void addFeature( std::shared_ptr< Feature >& feature );

    void removeFeature( std::shared_ptr< Feature >& feature );

    /// number of features
    std::size_t numberObservation() const;

    /// if the point is in the front of camera (z>0) and can projectable into image
    bool isVisible( const Eigen::Vector3d& point3D ) const;

    /// is this frame a keyframe
    bool isKeyframe() const;

    /// project from world to image pixel coordinate
    Eigen::Vector2d world2image( const Eigen::Vector3d& point3D_w ) const;

    /// project from world to camera coordinate
    Eigen::Vector3d world2camera( const Eigen::Vector3d& point3D_w ) const;

    /// project from camera coordinate to world
    Eigen::Vector3d camera2world( const Eigen::Vector3d& point3D_c ) const;

    /// project from camera coordinate to image pixel coordinate
    Eigen::Vector2d camera2image( const Eigen::Vector3d& point3D_c ) const;

    /// project from image pixel coordinate to world cooridnate
    Eigen::Vector3d image2world( const Eigen::Vector2d& point2D, const double depth ) const;

    /// project from image pixel cooridnate to camera coordinate
    Eigen::Vector3d image2camera( const Eigen::Vector2d& point2D, const double depth ) const;

    /// compute position of camera in world cordinate C = -R^t * t
    Eigen::Vector3d cameraInWorld() const;

    // TODO: reset frame

private:
};

#endif /* __FRAME_H__ */