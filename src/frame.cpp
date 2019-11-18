#include "frame.hpp"

Frame::Frame(Eigen::Matrix3d& K, cv::Mat& img): m_imagePyramid(img, 4)
{
    
}

void Frame::initFrame(const cv::Mat& img)
{
    if (img.empty())
        throw std::runtime_error("BANGO");
    
    m_imagePyramid.createImagePyramid(img, 4);
}

void Frame::setKeyframe()
{
    m_keyFrame = true;
}

void Frame::addFeature(Feature* feature)
{
    m_frameFeatures.push_back(feature);
}

void Frame::removeKeyPoint(Feature* feature)
{
    // std::remove_if(m_frameFeatures.begin(), m_frameFeatures.end(), [&feature](Feature*& f){if (f == feature)
    // {
    //     f = nullptr;
    // }});
    for(auto& f : m_frameFeatures)
    {
        if (f == feature)
        {
            f = nullptr;
            break;
        }
    }
}

std::uint32_t Frame::numberObservation() const
{
    return m_frameFeatures.size();
}

bool Frame::isVisible(const Eigen::Vector3d& point3D) const
{
    const Eigen::Vector3d cameraPoint = m_TransW2F * point3D;
    if (cameraPoint.z() < 0.0)
        return false;
    const Eigen::Vector2d imagePoint = camera2image(cameraPoint);
    if (imagePoint.x() >= 0.0 && imagePoint.y() >= 0.0 && imagePoint.x() < m_imgRes.second && imagePoint.y() < m_imgRes.first)
        return true;
    return false;
}

bool Frame::isKeyframe() const
{
    return m_keyFrame;
}


Eigen::Vector2d Frame::world2image (const Eigen::Vector3d& point3D_w) const
{
    const Eigen::Vector3d cameraPoint = world2camera(point3D_w);
    return camera2image(cameraPoint);
}


Eigen::Vector3d Frame::world2camera (const Eigen::Vector3d& point3D_w) const
{
    return m_TransW2F * point3D_w;
}


Eigen::Vector3d Frame::camera2world (const Eigen::Vector3d& point3D_c) const
{
    return m_TransW2F.inverse() * point3D_c;
}


Eigen::Vector2d Frame::camera2image (const Eigen::Vector3d& point3D_c) const
{
    const Eigen::Vector3d imagePoint = m_K * point3D_c;
    return Eigen::Vector2d(imagePoint.x()/imagePoint.z(), imagePoint.y()/imagePoint.z());
}


Eigen::Vector3d Frame::image2world (const Eigen::Vector2d& point2D, const double depth) const
{
    const Eigen::Vector3d cameraPoint = image2camera(point2D, depth);
    return camera2world(cameraPoint);
}


Eigen::Vector3d Frame::image2camera (const Eigen::Vector2d& point2D, const double depth) const
{
    const Eigen::Vector3d homoImageCord(point2D.x() * depth, point2D.y() * depth, depth);
    return m_K.inverse() * homoImageCord;
}

/// compute position of camera in world cordinate C = -R^t * t
Eigen::Vector3d Frame::cameraInWorld () const
{
    return m_TransW2F.inverse().translation();
}