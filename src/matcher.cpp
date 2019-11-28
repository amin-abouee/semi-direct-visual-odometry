#include "matcher.hpp"
#include <iostream>
#include <memory>

#include "feature.hpp"

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
// #include <opencv2/highgui/highgui.hpp>
// #include <opencv2/objdetect/objdetect.hpp>

bool Matcher::findEpipolarMatch(
  Frame& refFrame, Frame& curFrame, Feature& ft, const double minDepth, const double maxDepth, double& estimatedDepth )
{
    return false;
}

void Matcher::findTemplateMatch( Frame& refFrame,
                                 Frame& curFrame,
                                 const uint16_t patchSzRef,
                                 const uint16_t patchSzCur )
{
    // const std::uint32_t numFeature = refFrame.numberObservation();
    cv::Mat& refImg = refFrame.m_imagePyramid.getBaseImage();
    cv::Mat& curImg = curFrame.m_imagePyramid.getBaseImage();
    // std::cout << "ref type: " << refImg.type() << ", size: " << refImg.size() << std::endl;
    // std::cout << "cur type: " << curImg.type() << ", size: " << curImg.size() << std::endl;
    const uint16_t halfPatchRef = patchSzRef / 2;
    const uint16_t halfPatchCur = patchSzCur / 2;
    Eigen::Vector2i px( 0.0, 0.0 );
    const uint16_t offset = patchSzCur - patchSzRef;
    cv::Mat result( cv::Size( offset, offset ), CV_32F );

    double minVal, maxVal;
    cv::Point minLoc, maxLoc, matchLoc;
    // cv::Mat template (cv::Size(patchSize, patchSize), CV_32F);
    // cv::Mat  (cv::Size(patchSize, patchSize), CV_32F);
    for ( const auto& features : refFrame.m_frameFeatures )
    {
        px << features->m_feature.x(), features->m_feature.y();
        // std::cout << "px: " << px.transpose() << std::endl;
        // std::cout << "refFrame.m_camera->isInFrame: " << refFrame.m_camera->isInFrame( features->m_feature,
        // halfPatchRef ) << std::endl; std::cout << "curFrame.m_camera->isInFrame: " << curFrame.m_camera->isInFrame(
        // features->m_feature, halfPatchRef ) << std::endl;
        if ( refFrame.m_camera->isInFrame( features->m_feature, halfPatchRef ) &&
             curFrame.m_camera->isInFrame( features->m_feature, halfPatchCur ) )
        {
            // std::cout << "px is inside the image " << std::endl;
            cv::Rect ROITemplate( px.x() - halfPatchRef, px.y() - halfPatchRef, patchSzRef, patchSzRef );
            cv::Rect ROIImage( px.x() - halfPatchCur, px.y() - halfPatchCur, patchSzCur, patchSzCur );
            cv::Mat templatePatch = refImg( ROITemplate );
            cv::Mat imagePatch    = curImg( ROIImage );
            cv::matchTemplate( imagePatch, templatePatch, result, cv::TM_SQDIFF_NORMED );
            normalize( result, result, 0, 1, cv::NORM_MINMAX, -1, cv::Mat() );
            minMaxLoc( result, &minVal, &maxVal, &minLoc, &maxLoc, cv::Mat() );
            // matchLoc = minLoc;
            // std::cout << "corresponding loc: " << matchLoc << std::endl;
            Eigen::Vector2d newLoc;
            newLoc.x() = px.x() + offset + minLoc.x;
            newLoc.y() = px.y() + offset + minLoc.y;
            std::unique_ptr< Feature > newFeature = std::make_unique< Feature >(
              curFrame, newLoc, 0.0, 0.0, 0 );
            curFrame.addFeature( newFeature );
        }
    }
}

void Matcher::findOpticalFlowSparse(Frame& refFrame, Frame& curFrame, const uint16_t patchSize)
{
    std::vector <cv::Point2f> refPoints;
    std::vector <cv::Point2f> curPoints;
    for ( const auto& features : refFrame.m_frameFeatures )
        refPoints.push_back(cv::Point(features->m_feature.x(), features->m_feature.y()));

    std::vector<uchar> status;
    std::vector<float> err;

    const cv::Mat& refImg = refFrame.m_imagePyramid.getBaseImage();
    const cv::Mat& curImg = curFrame.m_imagePyramid.getBaseImage();
    cv::calcOpticalFlowPyrLK(refImg, curImg, refPoints, curPoints, status, err, cv::Size(patchSize, patchSize), 3);
    // uint32_t cnt = 0;
    for(std::size_t i(0); i< curPoints.size(); i++)
    {
        if (status[i])
        {
            // cnt++;
            std::unique_ptr< Feature > newFeature = std::make_unique< Feature >(
              curFrame, Eigen::Vector2d(curPoints[i].x, curPoints[i].y), 0.0, 0.0, 0 );
            curFrame.addFeature( newFeature );
        }
    }
    // std::cout << "cnt: " << cnt << std::endl;
}