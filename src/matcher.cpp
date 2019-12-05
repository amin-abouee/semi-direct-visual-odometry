#include "matcher.hpp"
#include <iostream>
#include <memory>

#include "feature.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
// #include <opencv2/highgui/highgui.hpp>
// #include <opencv2/objdetect/objdetect.hpp>

void Matcher::computeOpticalFlowSparse( Frame& refFrame, Frame& curFrame, const uint16_t patchSize )
{
    const cv::Mat& refImg = refFrame.m_imagePyramid.getBaseImage();
    const cv::Mat& curImg = curFrame.m_imagePyramid.getBaseImage();
    std::vector< cv::Point2f > refPoints;
    std::vector< cv::Point2f > curPoints;
    std::vector< uchar > status;
    std::vector< float > err;
    const int maxIteration    = 30;
    const double epsilonError = 1e-4;

    for ( const auto& features : refFrame.m_frameFeatures )
    {
        refPoints.emplace_back( cv::Point2f( static_cast< float >( features->m_feature.x() ),
                                             static_cast< float >( features->m_feature.y() ) ) );
        curPoints.emplace_back( cv::Point2f( static_cast< float >( features->m_feature.x() ),
                                             static_cast< float >( features->m_feature.y() ) ) );
    }

    // std::transform(
    //   refFrame.m_frameFeatures.cbegin(), refFrame.m_frameFeatures.cend(), std::back_inserter( refPoints ),
    //   []( const auto& features ) { return cv::Point2f( features->m_feature.x(), features->m_feature.y() ); } );

    cv::TermCriteria termcrit( cv::TermCriteria::COUNT + cv::TermCriteria::EPS, maxIteration, epsilonError );
    cv::calcOpticalFlowPyrLK( refImg, curImg, refPoints, curPoints, status, err, cv::Size( patchSize, patchSize ), 3,
                              termcrit, cv::OPTFLOW_USE_INITIAL_FLOW );

    for ( std::size_t i( 0 ); i < curPoints.size(); i++ )
    {
        if ( status[ i ] )
        {
            std::unique_ptr< Feature > newFeature = std::make_unique< Feature >(
              curFrame, Eigen::Vector2d( curPoints[ i ].x, curPoints[ i ].y ), 0.0, 0.0, 0 );
            curFrame.addFeature( newFeature );
        }
    }

    uint32_t cnt = 0;
    /// if status[i] == true, it have to return false because we dont want to remove it from our container
    auto isNotValid = [&cnt, &status, &refFrame]( const auto& feature ) {
        if ( feature->m_frame == &refFrame )
            return status[ cnt++ ] ? false : true;
        else
            return false;
    };

    // https://en.wikipedia.org/wiki/Erase%E2%80%93remove_idiom
    refFrame.m_frameFeatures.erase(
      std::remove_if( refFrame.m_frameFeatures.begin(), refFrame.m_frameFeatures.end(), isNotValid ),
      refFrame.m_frameFeatures.end() );

    // std::cout << "observation refFrame: " << refFrame.numberObservation() << std::endl;
    // std::cout << "observation curFrame: " << curFrame.numberObservation() << std::endl;
}

void Matcher::computeEssentialMatrix( Frame& refFrame, Frame& curFrame, const double reproError, Eigen::Matrix3d& E )
{
    std::vector< cv::Point2f > refPoints;
    std::vector< cv::Point2f > curPoints;
    std::vector< uchar > status;
    const std::size_t featureSize = refFrame.numberObservation();

    for ( std::size_t i( 0 ); i < featureSize; i++ )
    {
        refPoints.emplace_back( cv::Point2f( static_cast< float >( refFrame.m_frameFeatures[ i ]->m_feature.x() ),
                                             static_cast< float >( refFrame.m_frameFeatures[ i ]->m_feature.y() ) ) );
        curPoints.emplace_back( cv::Point2f( static_cast< float >( curFrame.m_frameFeatures[ i ]->m_feature.x() ),
                                             static_cast< float >( curFrame.m_frameFeatures[ i ]->m_feature.y() ) ) );
    }

    cv::Mat E_cv =
      cv::findEssentialMat( refPoints, curPoints, refFrame.m_camera->K_cv(), cv::RANSAC, 0.999, reproError, status );

    // std::cout << "type E_cv: " << E_cv.type() << std::endl;
    double* essential = E_cv.ptr< double >( 0 );
    E << essential[ 0 ], essential[ 1 ], essential[ 2 ], essential[ 3 ], essential[ 4 ], essential[ 5 ], essential[ 6 ],
      essential[ 7 ], essential[ 8 ];

    // std::cout << "E: " << E << std::endl;

    uint32_t cnt = 0;
    /// if status[i] == true, it have to return false because we dont want to remove it from our container
    auto isNotValidinRefFrame = [&cnt, &status, &refFrame]( const auto& feature ) {
        if ( feature->m_frame == &refFrame )
            return status[ cnt++ ] ? false : true;
        else
            return false;
    };

    // https://en.wikipedia.org/wiki/Erase%E2%80%93remove_idiom
    auto refResult =
      std::remove_if( refFrame.m_frameFeatures.begin(), refFrame.m_frameFeatures.end(), isNotValidinRefFrame );
    refFrame.m_frameFeatures.erase( refResult, refFrame.m_frameFeatures.end() );
    // std::cout << "observation refFrame: " << refFrame.numberObservation() << std::endl;

    auto isNotValidinCurFrame = [&cnt, &status, &curFrame]( const auto& feature ) {
        if ( feature->m_frame == &curFrame )
            return status[ cnt++ ] ? false : true;
        else
            return false;
    };
    cnt = 0;
    auto curResult =
      std::remove_if( curFrame.m_frameFeatures.begin(), curFrame.m_frameFeatures.end(), isNotValidinCurFrame );
    curFrame.m_frameFeatures.erase( curResult, curFrame.m_frameFeatures.end() );
    // std::cout << "observation curFrame: " << curFrame.numberObservation() << std::endl;
}

// bool Matcher::findEpipolarMatch(
//   Frame& refFrame, Frame& curFrame, Feature& ft, const double minDepth, const double maxDepth, double& estimatedDepth
//   )
// {
//     return false;
// }

void Matcher::templateMatching( const Frame& refFrame,
                                Frame& curFrame,
                                const uint16_t patchSzRef,
                                const uint16_t patchSzCur )
{
    // const std::uint32_t numFeature = refFrame.numberObservation();
    const cv::Mat& refImg = refFrame.m_imagePyramid.getBaseImage();
    const cv::Mat& curImg = curFrame.m_imagePyramid.getBaseImage();
    // std::cout << "ref type: " << refImg.type() << ", size: " << refImg.size() << std::endl;
    // std::cout << "cur type: " << curImg.type() << ", size: " << curImg.size() << std::endl;
    const uint16_t halfPatchRef = patchSzRef / 2;
    const uint16_t halfPatchCur = patchSzCur / 2;
    Eigen::Vector2i px( 0.0, 0.0 );
    const int32_t offset = patchSzCur - patchSzRef;
    cv::Mat result( cv::Size( offset, offset ), CV_32F );

    double minVal, maxVal;
    cv::Point2i minLoc, maxLoc;
    // cv::Mat template (cv::Size(patchSize, patchSize), CV_32F);
    // cv::Mat  (cv::Size(patchSize, patchSize), CV_32F);
    for ( const auto& features : refFrame.m_frameFeatures )
    {
        px << static_cast< int32_t >( features->m_feature.x() ), static_cast< int32_t >( features->m_feature.y() );
        // std::cout << "px: " << px.transpose() << std::endl;
        // std::cout << "refFrame.m_camera->isInFrame: " << refFrame.m_camera->isInFrame( features->m_feature,
        // halfPatchRef ) << std::endl; std::cout << "curFrame.m_camera->isInFrame: " << curFrame.m_camera->isInFrame(
        // features->m_feature, halfPatchRef ) << std::endl;
        if ( refFrame.m_camera->isInFrame( features->m_feature, halfPatchRef ) &&
             curFrame.m_camera->isInFrame( features->m_feature, halfPatchCur ) )
        {
            // std::cout << "px is inside the image " << std::endl;
            const cv::Rect ROITemplate( px.x() - halfPatchRef, px.y() - halfPatchRef, patchSzRef, patchSzRef );
            const cv::Rect ROIImage( px.x() - halfPatchCur, px.y() - halfPatchCur, patchSzCur, patchSzCur );
            const cv::Mat templatePatch = refImg( ROITemplate );
            const cv::Mat imagePatch    = curImg( ROIImage );
            cv::matchTemplate( imagePatch, templatePatch, result, cv::TM_SQDIFF_NORMED );
            normalize( result, result, 0, 1, cv::NORM_MINMAX, -1, cv::Mat() );
            minMaxLoc( result, &minVal, &maxVal, &minLoc, &maxLoc, cv::Mat() );
            // matchLoc = minLoc;
            // std::cout << "corresponding loc: " << matchLoc << std::endl;
            Eigen::Vector2d newLoc;
            newLoc.x()                            = static_cast< double >( px.x() + offset + minLoc.x );
            newLoc.y()                            = static_cast< double >( px.y() + offset + minLoc.y );
            std::unique_ptr< Feature > newFeature = std::make_unique< Feature >( curFrame, newLoc, 0.0, 0.0, 0 );
            curFrame.addFeature( newFeature );
        }
    }
}