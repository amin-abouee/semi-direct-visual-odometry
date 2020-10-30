#include "algorithm.hpp"
#include "feature.hpp"
#include "utils.hpp"

#include <algorithm>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/video/tracking.hpp>

#include "easylogging++.h"
#define Algorithm_Log( LEVEL ) CLOG( LEVEL, "Algorithm" )

// http://stackoverflow.com/questions/10167534/how-to-find-out-what-type-of-a-mat-object-is-with-mattype-in-opencv
// +--------+----+----+----+----+------+------+------+------+
// |        | C1 | C2 | C3 | C4 | C(5) | C(6) | C(7) | C(8) |
// +--------+----+----+----+----+------+------+------+------+
// | CV_8U  |  0 |  8 | 16 | 24 |   32 |   40 |   48 |   56 |
// | CV_8S  |  1 |  9 | 17 | 25 |   33 |   41 |   49 |   57 |
// | CV_16U |  2 | 10 | 18 | 26 |   34 |   42 |   50 |   58 |
// | CV_16S |  3 | 11 | 19 | 27 |   35 |   43 |   51 |   59 |
// | CV_32S |  4 | 12 | 20 | 28 |   36 |   44 |   52 |   60 |
// | CV_32F |  5 | 13 | 21 | 29 |   37 |   45 |   53 |   61 |
// | CV_64F |  6 | 14 | 22 | 30 |   38 |   46 |   54 |   62 |
// +--------+----+----+----+----+------+------+------+------+

// void algorithm::pointsRefCamera( const std::shared_ptr<Frame>& refFrame, const std::shared_ptr<Frame>& curFrame, Eigen::MatrixXd&
// pointsRefCamera )
// {
//     const auto featureSz = refFrame.numberObservation();
//     Eigen::Vector2d refFeature;
//     Eigen::Vector2d curFeature;
//     Eigen::Vector3d pointWorld;
//     for ( std::size_t i( 0 ); i < featureSz; i++ )
//     {
//         refFeature = refFrame.m_frameFeatures[ i ]->m_feature;
//         curFeature = curFrame.m_frameFeatures[ i ]->m_feature;
//         triangulatePointDLT( refFrame, curFrame, refFeature, curFeature, pointWorld );
//         pointsRefCamera.col( i ) = refFrame.world2camera( pointWorld );
//     }
// }

// void algorithm::pointsCurCamera( const std::shared_ptr<Frame>& refFrame, const std::shared_ptr<Frame>& curFrame, Eigen::MatrixXd&
// pointsCurCamera )
// {
//     const auto featureSz = refFrame.numberObservation();
//     Eigen::Vector2d refFeature;
//     Eigen::Vector2d curFeature;
//     Eigen::Vector3d pointWorld;
//     for ( std::size_t i( 0 ); i < featureSz; i++ )
//     {
//         refFeature = refFrame.m_frameFeatures[ i ]->m_feature;
//         curFeature = curFrame.m_frameFeatures[ i ]->m_feature;
//         triangulatePointDLT( refFrame, curFrame, refFeature, curFeature, pointWorld );
//         pointsCurCamera.col( i ) = curFrame.world2camera( pointWorld );
//     }
// }

void algorithm::computeOpticalFlowSparse( std::shared_ptr< Frame >& refFrame, std::shared_ptr< Frame >& curFrame, const uint32_t patchSize )
{
    const cv::Mat& refImg = refFrame->m_imagePyramid.getBaseImage();
    const cv::Mat& curImg = curFrame->m_imagePyramid.getBaseImage();
    std::vector< cv::Point2f > refPoints;
    std::vector< cv::Point2f > curPoints;
    std::vector< uchar > status;
    std::vector< float > err;
    const int maxIteration    = 30;
    const double epsilonError = 1e-4;

    for ( const auto& features : refFrame->m_frameFeatures )
    {
        refPoints.emplace_back(
          cv::Point2f( static_cast< float >( features->m_feature.x() ), static_cast< float >( features->m_feature.y() ) ) );
        curPoints.emplace_back(
          cv::Point2f( static_cast< float >( features->m_feature.x() ), static_cast< float >( features->m_feature.y() ) ) );
    }

    // std::transform(
    //   refFrame.m_frameFeatures.cbegin(), refFrame.m_frameFeatures.cend(), std::back_inserter( refPoints ),
    //   []( const auto& features ) { return cv::Point2f( features->m_feature.x(), features->m_feature.y() ); } );

    cv::TermCriteria termcrit( cv::TermCriteria::COUNT + cv::TermCriteria::EPS, maxIteration, epsilonError );
    cv::calcOpticalFlowPyrLK( refImg, curImg, refPoints, curPoints, status, err, cv::Size( patchSize, patchSize ), 3, termcrit,
                              cv::OPTFLOW_USE_INITIAL_FLOW );

    for ( std::size_t i( 0 ); i < curPoints.size(); i++ )
    {
        if ( status[ i ] == true )
        {
            std::shared_ptr< Feature > newFeature =
              std::make_shared< Feature >( curFrame, Eigen::Vector2d( curPoints[ i ].x, curPoints[ i ].y ), 0.0, 0.0, 0 );
            curFrame->addFeature( newFeature );
        }
    }

    uint32_t cnt = 0;
    /// if status[i] == true, it have to return false because we dont want to remove it from our container
    auto isNotValid = [ &cnt, &status ]( const auto& feature ) { return status[ cnt++ ] ? false : true; };

    // https://en.wikipedia.org/wiki/Erase%E2%80%93remove_idiom
    refFrame->m_frameFeatures.erase( std::remove_if( refFrame->m_frameFeatures.begin(), refFrame->m_frameFeatures.end(), isNotValid ),
                                     refFrame->m_frameFeatures.end() );

    Algorithm_Log( DEBUG ) << "observation refFrame: " << refFrame->numberObservation();
    Algorithm_Log( DEBUG ) << "observation curFrame: " << curFrame->numberObservation();
}

void algorithm::computeEssentialMatrix( std::shared_ptr< Frame >& refFrame,
                                        std::shared_ptr< Frame >& curFrame,
                                        const double reproError,
                                        Eigen::Matrix3d& E )
{
    std::vector< cv::Point2f > refPoints;
    std::vector< cv::Point2f > curPoints;
    std::vector< uchar > status;
    const std::size_t featureSize = refFrame->numberObservation();

    for ( std::size_t i( 0 ); i < featureSize; i++ )
    {
        refPoints.emplace_back( cv::Point2f( static_cast< float >( refFrame->m_frameFeatures[ i ]->m_feature.x() ),
                                             static_cast< float >( refFrame->m_frameFeatures[ i ]->m_feature.y() ) ) );
        curPoints.emplace_back( cv::Point2f( static_cast< float >( curFrame->m_frameFeatures[ i ]->m_feature.x() ),
                                             static_cast< float >( curFrame->m_frameFeatures[ i ]->m_feature.y() ) ) );
    }

    cv::Mat E_cv = cv::findEssentialMat( refPoints, curPoints, refFrame->m_camera->K_cv(), cv::RANSAC, 0.999, reproError, status );

    // std::cout << "type E_cv: " << E_cv.type() << std::endl;
    double* essential = E_cv.ptr< double >( 0 );
    E << essential[ 0 ], essential[ 1 ], essential[ 2 ], essential[ 3 ], essential[ 4 ], essential[ 5 ], essential[ 6 ], essential[ 7 ],
      essential[ 8 ];

    // std::cout << "E: " << E << std::endl;

    uint32_t cnt = 0;
    /// if status[i] == true, it have to return false because we dont want to remove it from our container
    auto isNotValidRefFrame = [ &cnt, &status, &refFrame ]( const auto& feature ) {
        if ( feature->m_frame == refFrame )
            return status[ cnt++ ] ? false : true;
        else
            return false;
    };

    // https://en.wikipedia.org/wiki/Erase%E2%80%93remove_idiom
    auto refResult = std::remove_if( refFrame->m_frameFeatures.begin(), refFrame->m_frameFeatures.end(), isNotValidRefFrame );
    refFrame->m_frameFeatures.erase( refResult, refFrame->m_frameFeatures.end() );
    // std::cout << "observation refFrame: " << refFrame.numberObservation() << std::endl;

    auto isNotValidCurFrame = [ &cnt, &status, &curFrame ]( const auto& feature ) {
        if ( feature->m_frame == curFrame )
            return status[ cnt++ ] ? false : true;
        else
            return false;
    };
    cnt            = 0;
    auto curResult = std::remove_if( curFrame->m_frameFeatures.begin(), curFrame->m_frameFeatures.end(), isNotValidCurFrame );
    curFrame->m_frameFeatures.erase( curResult, curFrame->m_frameFeatures.end() );
    // std::cout << "observation curFrame: " << curFrame.numberObservation() << std::endl;
}

void algorithm::templateMatching( const std::shared_ptr< Frame >& refFrame,
                                  std::shared_ptr< Frame >& curFrame,
                                  const uint16_t patchSzRef,
                                  const uint16_t patchSzCur )
{
    // const std::uint32_t numFeature = refFrame.numberObservation();
    const cv::Mat& refImg = refFrame->m_imagePyramid.getBaseImage();
    const cv::Mat& curImg = curFrame->m_imagePyramid.getBaseImage();
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
    for ( const auto& features : refFrame->m_frameFeatures )
    {
        px << static_cast< int32_t >( features->m_feature.x() ), static_cast< int32_t >( features->m_feature.y() );
        // std::cout << "px: " << px.transpose() << std::endl;
        // std::cout << "refFrame.m_camera->isInFrame: " << refFrame.m_camera->isInFrame( features->m_feature,
        // halfPatchRef ) << std::endl; std::cout << "curFrame.m_camera->isInFrame: " << curFrame.m_camera->isInFrame(
        // features->m_feature, halfPatchRef ) << std::endl;
        if ( refFrame->m_camera->isInFrame( features->m_feature, halfPatchRef ) &&
             curFrame->m_camera->isInFrame( features->m_feature, halfPatchCur ) )
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
            std::shared_ptr< Feature > newFeature = std::make_shared< Feature >( curFrame, newLoc, 0.0, 0.0, 0 );
            curFrame->addFeature( newFeature );
        }
    }
}

void algorithm::getAffineWarp( const std::shared_ptr< Frame >& refFrame,
                               const std::shared_ptr< Frame >& curFrame,
                               const std::shared_ptr< Feature >& feature,
                               const Sophus::SE3d& relativePose,
                               const uint32_t patchSize,
                               const double depth,
                               //    const int level,
                               Eigen::Matrix2d& affineWarp )
{
    const uint32_t halfPatchSize = patchSize / 2;

    // .--patch_size-->                 .-------------->
    // |-------------->                 |-----------u'->    ==> duDiff = u' - c'
    // |-------------->                 |-------------->
    // |------c-------u     ====>       |---c'--------->
    // |-------------->                 |-------------->
    // |-------------->                 |-------------->    ==> dvDiff = v' - c'
    // |------v------->                 |--------v'---->

    const Eigen::Vector3d centerRefCamera = refFrame->image2camera( feature->m_feature, depth );
    const Eigen::Vector3d duRefCamera     = refFrame->image2camera( feature->m_feature + Eigen::Vector2d( halfPatchSize, 0.0 ), depth );
    const Eigen::Vector3d dvRefCamera     = refFrame->image2camera( feature->m_feature + Eigen::Vector2d( 0.0, halfPatchSize ), depth );

    const Eigen::Vector2d centerCurImg = curFrame->camera2image( relativePose * centerRefCamera );
    const Eigen::Vector2d duCurImg     = curFrame->camera2image( relativePose * duRefCamera );
    const Eigen::Vector2d dvCurImg     = curFrame->camera2image( relativePose * dvRefCamera );

    const Eigen::Vector2d duDiff = duCurImg - centerCurImg;
    const Eigen::Vector2d dvDiff = dvCurImg - centerCurImg;

    affineWarp.col( 0 ) = duDiff / halfPatchSize;
    affineWarp.col( 1 ) = dvDiff / halfPatchSize;
}

void algorithm::applyAffineWarp( const std::shared_ptr< Frame >& frame,
                                 const Eigen::Vector2d& location,
                                 const int32_t halfPatchSize,
                                 const Eigen::Matrix2d& affineWarp,
                                 const double boundary,
                                 Eigen::Matrix< uint8_t, Eigen::Dynamic, 1 >& data )
{
    const cv::Mat& img = frame->m_imagePyramid.getBaseImage();
    const algorithm::MapXRowConst imgEigen( img.ptr< uint8_t >(), img.rows, img.cols );

    uint32_t idx = 0;
    if ( frame->m_camera->isInFrame( location, boundary ) )
    {
        for ( int32_t i( -halfPatchSize ); i <= halfPatchSize; i++ )
        {
            for ( int32_t j( -halfPatchSize ); j <= halfPatchSize; j++ )
            {
                const Eigen::Vector2d locationInPatch = location + affineWarp * Eigen::Vector2d( j, i );
                data( idx++ ) = algorithm::bilinearInterpolation( imgEigen, locationInPatch.x(), locationInPatch.y() );
            }
        }
    }
    // Algorithm_Log( DEBUG ) << "number of matched pixels: " << idx;
}

double algorithm::computeScore( const Eigen::Matrix< uint8_t, Eigen::Dynamic, 1 >& refPatchIntensity,
                                const Eigen::Matrix< uint8_t, Eigen::Dynamic, 1 >& curPatchIntensity )

{
    const double refMeanPatchInt = refPatchIntensity.mean();
    const double curMeanPatchInt = curPatchIntensity.mean();
    double sum                   = 0.0;
    for ( int32_t i( 0 ); i < refPatchIntensity.size(); i++ )
    {
        double error = ( refPatchIntensity( i ) - refMeanPatchInt ) - ( curPatchIntensity( i ) - curMeanPatchInt );
        // sum += error * error;
        sum += std::fabs( error );
    }
    return sum;
}

bool algorithm::matchDirect( const std::shared_ptr< Point >& point, const std::shared_ptr< Frame >& curFrame, Eigen::Vector2d& featurePose )
{
    const uint32_t patchSize     = 7;
    const uint32_t halfPatchSize = patchSize / 2;
    const uint32_t patchArea     = patchSize * patchSize;
    std::shared_ptr< Feature > feature;

    if ( point->getCloseViewObservation( curFrame->cameraInWorld(), feature ) == false )
        return false;

    if ( feature->m_frame->m_camera->isInFrame( feature->m_feature, halfPatchSize ) == false )
        return false;

    const Sophus::SE3d relativePose = algorithm::computeRelativePose( feature->m_frame, curFrame );
    const double depth              = ( feature->m_frame->cameraInWorld() - point->m_position ).norm();
    Eigen::Matrix2d affineWarp;
    algorithm::getAffineWarp( feature->m_frame, curFrame, feature, relativePose, 7, depth, affineWarp );

    Eigen::Matrix< uint8_t, Eigen::Dynamic, 1 > refPatchIntensities( patchArea );
    algorithm::applyAffineWarp( feature->m_frame, feature->m_feature, halfPatchSize, Eigen::Matrix2d::Identity(), halfPatchSize + 1,
                                refPatchIntensities );

    // TODO: optimize the pose
    // if (i = 0)
    // {
    return true;
    // }
    return false;
}

bool algorithm::matchEpipolarConstraint( const std::shared_ptr< Frame >& refFrame,
                                         const std::shared_ptr< Frame >& curFrame,
                                         std::shared_ptr< Feature >& refFeature,
                                         const uint32_t patchSize,
                                         const double initialDepth,
                                         const double minDepth,
                                         const double maxDepth,
                                         double& estimatedDepth )
{
    // const uint32_t patchSize     = 7;
    const uint32_t halfPatchSize = patchSize / 2;
    const uint32_t patchArea     = patchSize * patchSize;

    const Sophus::SE3d relativePose = algorithm::computeRelativePose( refFrame, curFrame );
    const uint32_t thresholdZSSD    = patchArea * 128;
    const Eigen::Vector2d locationCenter =
      curFrame->camera2image( relativePose * refFrame->image2camera( refFeature->m_feature, initialDepth ) );
    const Eigen::Vector2d locationMin = curFrame->camera2image( relativePose * refFrame->image2camera( refFeature->m_feature, minDepth ) );
    const Eigen::Vector2d locationMax = curFrame->camera2image( relativePose * refFrame->image2camera( refFeature->m_feature, maxDepth ) );

    // FIXME: check with the original code. They don't project to the camera. Just project2d (x/z, y/z)
    const Eigen::Vector2d epipolarDirection = locationMax - locationMin;

    Eigen::Matrix2d affineWarp;
    algorithm::getAffineWarp( refFrame, curFrame, refFeature, relativePose, patchSize, initialDepth, affineWarp );
    double normEpipolar = epipolarDirection.norm();
    Algorithm_Log( DEBUG ) << "affine matrix: " << affineWarp.format( utils::eigenFormat() );

    Eigen::Matrix< uint8_t, Eigen::Dynamic, 1 > refPatchIntensities( patchArea );
    refPatchIntensities.setZero();
    algorithm::applyAffineWarp( refFrame, refFeature->m_feature, halfPatchSize, Eigen::Matrix2d::Identity(), halfPatchSize + 1,
                                refPatchIntensities );

    if ( normEpipolar < 2.0 )
    {
        // TODO: 2D alignment
        const Eigen::Vector2d centerLocation   = ( locationMax + locationMin ) / 2.0;
        const Eigen::Vector3d bearingCurCamera = curFrame->m_camera->inverseProject2d( centerLocation );
        if ( depthFromTriangulation( relativePose, refFeature->m_bearingVec, bearingCurCamera, estimatedDepth ) )
        {
            Algorithm_Log( DEBUG ) << "pre depth: " << initialDepth << ", depth updated: " << estimatedDepth;
            return true;
        }
        else
        {
            return false;
        }
    }

    // Find length of search range on epipolar line
    // Vector2d px_A(cur_frame.cam_->world2cam(A));
    // Vector2d px_B(cur_frame.cam_->world2cam(B));
    // epi_length_ = (px_A-px_B).norm() / (1<<search_level_);

    const Eigen::Vector2d borders = affineWarp * Eigen::Vector2d( halfPatchSize, halfPatchSize );
    const double boundary         = std::max( borders.x(), borders.y() ) + 1;

    Eigen::Matrix< uint8_t, Eigen::Dynamic, 1 > curPatchIntensities( patchArea );
    curPatchIntensities.setZero();
    // TODO: why divide by 0.7. divide by norm
    // const uint32_t pixelStep           = normEpipolar / 0.7;
    // const Eigen::Vector2d step2D = epipolarDirection / pixelStep;
    const uint32_t pixelStep     = std::ceil( normEpipolar );
    const Eigen::Vector2d step2D = epipolarDirection / normEpipolar;
    double minimumScore          = std::numeric_limits< double >::max();
    Eigen::Vector2d bestLocation;

    Algorithm_Log( DEBUG ) << "Epipolar Direction: " << epipolarDirection.transpose() << ", norm: " << normEpipolar
                           << ", Pixel step: " << pixelStep << ", Step 2D: " << step2D.transpose();
    // Algorithm_Log( INFO ) << "Location in ref frame: " << refFeature->m_feature.transpose();
    // Algorithm_Log( INFO ) << "Ref patch intensities: " << refPatchIntensities.cast<int32_t>().transpose();

    // traverse over the epipolar line
    for ( uint32_t i( 0 ); i < pixelStep; i++ )
    {
        const Eigen::Vector2d location = locationMin + i * step2D;
        algorithm::applyAffineWarp( curFrame, location, halfPatchSize, affineWarp, boundary, curPatchIntensities );
        double zssd = computeScore( refPatchIntensities, curPatchIntensities );
        // Algorithm_Log( DEBUG ) << "Loc in cur frame: " << location.transpose() << ", zzd score: " << zssd;
        // Algorithm_Log( DEBUG ) << "Cur patch intensities: " << curPatchIntensities.cast<int32_t>().transpose();
        if ( zssd < minimumScore )
        {
            minimumScore = zssd;
            bestLocation = location;
        }
    }

    Algorithm_Log( DEBUG ) << "Min zssd score: " << minimumScore;
    if ( minimumScore < thresholdZSSD )
    {
        // TODO: 2D alignment
        const Eigen::Vector3d bearingCurCamera = curFrame->m_camera->inverseProject2d( bestLocation );
        Algorithm_Log( DEBUG ) << "pre location: " << bestLocation.transpose() << ", cur location: " << locationCenter.transpose();

        if ( depthFromTriangulation( relativePose, refFeature->m_bearingVec, bearingCurCamera, estimatedDepth ) )
        {
            Algorithm_Log( DEBUG ) << "pre depth: " << initialDepth << ", depth updated: " << estimatedDepth;
            return true;
        }
    }
    else
    {
        return false;
    }
}

void algorithm::triangulate3DWorldPoints( const std::shared_ptr< Frame >& refFrame,
                                          const std::shared_ptr< Frame >& curFrame,
                                          Eigen::MatrixXd& pointsWorld )
{
    const auto featureSz = refFrame->numberObservation();
    Eigen::Vector3d pointWorld;
    for ( std::size_t i( 0 ); i < featureSz; i++ )
    {
        const Eigen::Vector2d refFeature = refFrame->m_frameFeatures[ i ]->m_feature;
        const Eigen::Vector2d curFeature = curFrame->m_frameFeatures[ i ]->m_feature;
        triangulatePointDLT( refFrame, curFrame, refFeature, curFeature, pointWorld );
        pointsWorld.col( i ) = pointWorld;
    }
}

void algorithm::transferPointsWorldToCam( const std::shared_ptr< Frame >& frame,
                                          const Eigen::MatrixXd& pointsWorld,
                                          Eigen::MatrixXd& pointsCamera )
{
    const auto featureSz = frame->numberObservation();
    for ( std::size_t i( 0 ); i < featureSz; i++ )
    {
        pointsCamera.col( i ) = frame->world2camera( pointsWorld.col( i ) );
    }
}

void algorithm::transferPointsCamToWorld( const std::shared_ptr< Frame >& frame,
                                          const Eigen::MatrixXd& pointsCamera,
                                          Eigen::MatrixXd& pointsWorld )
{
    const auto featureSz = frame->numberObservation();
    for ( std::size_t i( 0 ); i < featureSz; i++ )
    {
        pointsWorld.col( i ) = frame->camera2world( pointsCamera.col( i ) );
    }
}

void algorithm::normalizedDepthCamera( const std::shared_ptr< Frame >& frame,
                                       const Eigen::MatrixXd& pointsWorld,
                                       Eigen::VectorXd& normalizedDepthCamera )
{
    const auto featureSz = frame->numberObservation();
    for ( std::size_t i( 0 ); i < featureSz; i++ )
    {
        normalizedDepthCamera( i ) = frame->world2camera( pointsWorld.col( i ) ).norm();
    }
}

void algorithm::normalizedDepthCamera( const std::shared_ptr< Frame >& frame, Eigen::VectorXd& normalizedDepthCamera )
{
    const auto featureSz = frame->numberObservation();
    for ( std::size_t i( 0 ); i < featureSz; i++ )
    {
        normalizedDepthCamera( i ) = frame->world2camera( frame->m_frameFeatures[ i ]->m_point->m_position ).norm();
    }
}

void algorithm::depthCamera( const std::shared_ptr< Frame >& frame, const Eigen::MatrixXd& pointsWorld, Eigen::VectorXd& depthCamera )
{
    const auto featureSz = frame->numberObservation();
    for ( std::size_t i( 0 ); i < featureSz; i++ )
    {
        depthCamera( i ) = frame->world2camera( pointsWorld.col( i ) ).z();
    }
}

void algorithm::depthCamera( const std::shared_ptr< Frame >& frame, Eigen::VectorXd& depthCamera )
{
    const auto featureSz = frame->numberObservation();
    for ( std::size_t i( 0 ); i < featureSz; i++ )
    {
        depthCamera( i ) = frame->world2camera( frame->m_frameFeatures[ i ]->m_point->m_position ).z();
    }
}

void algorithm::triangulatePointHomogenousDLT( const std::shared_ptr< Frame >& refFrame,
                                               const std::shared_ptr< Frame >& curFrame,
                                               const Eigen::Vector2d& refFeature,
                                               const Eigen::Vector2d& curFeature,
                                               Eigen::Vector3d& point )
{
    Eigen::MatrixXd A( 4, 4 );
    const Eigen::Matrix< double, 3, 4 > P1 = refFrame->m_camera->K() * refFrame->m_TransW2F.matrix3x4();
    const Eigen::Matrix< double, 3, 4 > P2 = curFrame->m_camera->K() * curFrame->m_TransW2F.matrix3x4();

    A.row( 0 ) = ( refFeature.x() * P1.row( 2 ) ) - P1.row( 0 );
    A.row( 1 ) = ( refFeature.y() * P1.row( 2 ) ) - P1.row( 1 );
    A.row( 2 ) = ( curFeature.x() * P2.row( 2 ) ) - P2.row( 0 );
    A.row( 3 ) = ( curFeature.y() * P2.row( 2 ) ) - P2.row( 1 );

    // A.row(0) = refFeature.y() * P1.row(2) - P1.row(1);
    // A.row(1) = P1.row(0) - refFeature.x() * P1.row(2);
    // A.row(0) = curFeature.y() * P2.row(2) - P2.row(1);
    // A.row(1) = P2.row(0) - curFeature.x() * P2.row(2);

    A.row( 0 ) /= A.row( 0 ).norm();
    A.row( 1 ) /= A.row( 1 ).norm();
    A.row( 2 ) /= A.row( 2 ).norm();
    A.row( 3 ) /= A.row( 3 ).norm();

    Eigen::JacobiSVD< Eigen::MatrixXd > svd_A( A.transpose() * A, Eigen::ComputeFullV );
    Eigen::VectorXd res = svd_A.matrixV().col( 3 );
    res /= res.w();

    // Eigen::Vector2d project1 = refFrame.world2image( res.head( 2 ) );
    // Eigen::Vector2d project2 = curFrame.world2image( res.head( 2 ) );
    // std::cout << "Error in ref: " << ( project1 - refFeature ).norm()
    //           << ", Error in cur: " << ( project2 - curFeature ).norm() << std::endl;
    // std::cout << "project 1: " << project1.transpose() << std::endl;
    // std::cout << "project 2: " << project2.transpose() << std::endl;
    // std::cout << "point in reference camera: " << refFrame.world2camera( res.head( 2 ) ).transpose() << std::endl;
    // std::cout << "point in current camera: " << curFrame.world2camera( res.head( 2 ) ).transpose() << std::endl;
    point = res.head( 2 );
}

void algorithm::triangulatePointDLT( const std::shared_ptr< Frame >& refFrame,
                                     const std::shared_ptr< Frame >& curFrame,
                                     const Eigen::Vector2d& refFeature,
                                     const Eigen::Vector2d& curFeature,
                                     Eigen::Vector3d& point )
{
    Eigen::MatrixXd A( 4, 3 );
    const Eigen::Matrix< double, 3, 4 > P1 = refFrame->m_camera->K() * refFrame->m_TransW2F.matrix3x4();
    const Eigen::Matrix< double, 3, 4 > P2 = curFrame->m_camera->K() * curFrame->m_TransW2F.matrix3x4();
    // Algorithm_Log(DEBUG) << "pose reference: " << curFrame->m_TransW2F.params().transpose();

    A.row( 0 ) << P1( 0, 0 ) - refFeature.x() * P1( 2, 0 ), P1( 0, 1 ) - refFeature.x() * P1( 2, 1 ),
      P1( 0, 2 ) - refFeature.x() * P1( 2, 2 );
    A.row( 1 ) << P1( 1, 0 ) - refFeature.y() * P1( 2, 0 ), P1( 1, 1 ) - refFeature.y() * P1( 2, 1 ),
      P1( 1, 2 ) - refFeature.y() * P1( 2, 2 );
    A.row( 2 ) << P2( 0, 0 ) - curFeature.x() * P2( 2, 0 ), P2( 0, 1 ) - curFeature.x() * P2( 2, 1 ),
      P2( 0, 2 ) - curFeature.x() * P2( 2, 2 );
    A.row( 3 ) << P2( 1, 0 ) - curFeature.y() * P2( 2, 0 ), P2( 1, 1 ) - curFeature.y() * P2( 2, 1 ),
      P2( 1, 2 ) - curFeature.y() * P2( 2, 2 );

    Eigen::VectorXd p( 4 );
    p << refFeature.x() * P1( 2, 3 ) - P1( 0, 3 ), refFeature.y() * P1( 2, 3 ) - P1( 1, 3 ), curFeature.x() * P2( 2, 3 ) - P2( 0, 3 ),
      curFeature.y() * P2( 2, 3 ) - P2( 1, 3 );
    // point = A.colPivHouseholderQr().solve(p);
    point = ( A.transpose() * A ).ldlt().solve( A.transpose() * p );

    /*
    Eigen::Vector2d project1 = refFrame.world2image( point );
    Eigen::Vector2d project2 = curFrame.world2image( point );
    // std::cout << "Pt -> ref: " << refFeature.transpose() << ", cur: " << curFeature.transpose() << std::endl;
    // std::cout << "2D -> Error in ref: " << ( project1 - refFeature ).norm()
    //   << ", Error in cur: " << ( project2 - curFeature ).norm() << std::endl;

    Eigen::Vector3d unproject1 = refFrame.image2camera( refFeature, point.norm() );
    Eigen::Vector3d unproject2 = curFrame.image2camera( curFeature, point.norm() );
    // std::cout << "3D -> Error in ref: " << (point - unproject1).norm()
    //   << ", Error in cur: " << (point - unproject2).norm() << std::endl;

    Sophus::SE3d T_pre_cur      = refFrame.m_TransW2F.inverse() * curFrame.m_TransW2F;
    Eigen::Vector3d transferred = T_pre_cur * unproject1;
    // std::cout << "3D -> Error in relative: " << (transferred - unproject2).norm() << std::endl;
    // std::cout << "2D -> Error in relative: " << ( curFeature - curFrame.camera2image( transferred ) ).norm()
    std::cout << "Pt ref: " << refFeature.transpose()
              << ", error: " << ( curFeature - curFrame.camera2image( transferred ) ).norm()
              << ", depth: " << point.norm() << std::endl;
    */
}

bool algorithm::depthFromTriangulation( const Sophus::SE3d& relativePose,
                                        const Eigen::Vector3d& refBearingVec,
                                        const Eigen::Vector3d& curBearingVec,
                                        double& depth )
{
    // R * (bea_ref * d1) + t = bea_cur * d2
    // R * bea_ref * d1 - bea_cur * d2 = -t
    // [R * bea_ref, -bea_cur][d1; d2] = -t

    Eigen::Matrix< double, 3, 2 > A;
    A << relativePose.rotationMatrix() * refBearingVec, -curBearingVec;
    const Eigen::Matrix2d AtA = A.transpose() * A;
    if ( AtA.determinant() < 0.000001 )
    {
        return false;
    }
    const Eigen::Vector2d depths = -AtA.inverse() * A.transpose() * relativePose.translation();
    Algorithm_Log( DEBUG ) << "depths: " << depths.transpose();
    depth = std::fabs( depths.x() );
    return true;
}

// 9.6.2 Extraction of cameras from the essential matrix, multi view geometry
// https://github.com/opencv/opencv/blob/a74fe2ec01d9218d06cb7675af633fc3f409a6a2/modules/calib3d/src/five-point.cpp
void algorithm::decomposeEssentialMatrix( const Eigen::Matrix3d& E, Eigen::Matrix3d& R1, Eigen::Matrix3d& R2, Eigen::Vector3d& t )
{
    Eigen::JacobiSVD< Eigen::Matrix3d > svd_E( E, Eigen::ComputeFullV | Eigen::ComputeFullU );
    Eigen::Matrix3d W;
    W << 0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0;
    R1 = svd_E.matrixU() * W * svd_E.matrixV().transpose();
    if ( R1.determinant() < 0 )
        R1 *= -1;
    // std::cout << "R1: " << R1.format( utils::eigenFormat() ) << std::endl;

    R2 = svd_E.matrixU() * W.transpose() * svd_E.matrixV().transpose();
    if ( R2.determinant() < 0 )
        R2 *= -1;
    // std::cout << "R2: " << R2.format( utils::eigenFormat() ) << std::endl;

    t = svd_E.matrixU().col( 2 );
    // std::cout << "t: " << t.format( utils::eigenFormat() ) << std::endl;

    // cv::Mat R2c;
    // cv::Mat R1c;
    // cv::Mat tc;
    // cv::Mat Ec;
    // cv::eigen2cv(E, Ec);
    // cv::decomposeEssentialMat(Ec, R1c, R2c, tc);
    // std::cout << "R1: " << R1c << std::endl;
    // std::cout << "R2: " << R2c << std::endl;
    // std::cout << "t: " << tc << std::endl;
}

void algorithm::recoverPose( const Eigen::Matrix3d& E,
                             const std::shared_ptr< Frame >& refFrame,
                             std::shared_ptr< Frame >& curFrame,
                             Eigen::Matrix3d& R,
                             Eigen::Vector3d& t )
{
    Eigen::Matrix3d R1;
    Eigen::Matrix3d R2;
    Eigen::Vector3d tm;
    decomposeEssentialMatrix( E, R1, R2, tm );
    // std::cout << "R1: " << R1.format( utils::eigenFormat() ) << std::endl;
    // std::cout << "R2: " << R2.format( utils::eigenFormat() ) << std::endl;
    // std::cout << "t: " << t.format( utils::eigenFormat() ) << std::endl;

    // Eigen::Vector2d topLeftCorner(1.0, 1.0);
    // Eigen::Vector2d downRightCorner(refFrame.m_camera->width(), refFrame.m_camera->height());

    std::vector< Sophus::SE3d, Eigen::aligned_allocator< Sophus::SE3d > > poses;
    poses.reserve( 4 );
    Eigen::AngleAxisd temp( R1 );  // Re-orthogonality
    poses.emplace_back( Sophus::SE3d( temp.toRotationMatrix(), tm ) );
    poses.emplace_back( Sophus::SE3d( temp.toRotationMatrix(), -tm ) );
    temp = Eigen::AngleAxisd( R2 );
    poses.emplace_back( Sophus::SE3d( temp.toRotationMatrix(), tm ) );
    poses.emplace_back( Sophus::SE3d( temp.toRotationMatrix(), -tm ) );

    for ( std::size_t i( 0 ); i < 4; i++ )
    {
        Eigen::Vector3d point1;
        Eigen::Vector3d point2;

        // ^{K}_{W}T = ^{K}_{K-1}T \, * \,^{K-1}_{W}T
        curFrame->m_TransW2F = poses[ i ] * refFrame->m_TransW2F;
        // Algorithm_Log(DEBUG) << "pose reference 1: " << curFrame->m_TransW2F.params().transpose();
        // Algorithm_Log(DEBUG) << "pose reference 2: " << (poses[i] * refFrame->m_TransW2F).params().transpose();

        triangulatePointDLT( refFrame, curFrame, refFrame->m_frameFeatures[ 0 ]->m_feature, curFrame->m_frameFeatures[ 0 ]->m_feature,
                             point1 );
        triangulatePointDLT( refFrame, curFrame, refFrame->m_frameFeatures[ 1 ]->m_feature, curFrame->m_frameFeatures[ 1 ]->m_feature,
                             point2 );
        Eigen::Vector3d refProject1 = refFrame->world2camera( point1 );
        Eigen::Vector3d curProject1 = curFrame->world2camera( point1 );
        Eigen::Vector3d refProject2 = refFrame->world2camera( point2 );
        Eigen::Vector3d curProject2 = curFrame->world2camera( point2 );
        // std::cout << "output projct left corner, ref: " << refProject1.z() << ", cur: " << curProject1.z() <<
        // std::endl; std::cout << "output projct right corner, ref: " << refProject2.z() << ", cur: " <<
        // curProject2.z() << std::endl;
        if ( refProject1.z() > 0 && refProject2.z() > 0 && curProject1.z() > 0 && curProject2.z() > 0 )
        {
            R = poses[ i ].rotationMatrix();
            t = poses[ i ].translation();
            // std::cout << "R: " << R.format( utils::eigenFormat() ) << std::endl;
            // std::cout << "t: " << t.format( utils::eigenFormat() ) << std::endl;
            break;
        }
    }
}

Sophus::SE3d algorithm::computeRelativePose( const std::shared_ptr< Frame >& refFrame, const std::shared_ptr< Frame >& curFrame )
{
    // T_K-1_K = T_K-1_W * T_W_K
    // return refFrame->m_TransW2F.inverse() * curFrame->m_TransW2F;

    // ^{K}_{K-1}T = ^{K}_{W}T \, * \,^{W}_{K-1}T = ^{K}_{W}T \, * \,^{K-1}_{W}T^{-1}
    return curFrame->m_TransW2F * refFrame->m_TransW2F.inverse();
}

// bool algorithm::checkCheirality()
// {
//     return true;
// }

Eigen::Matrix3d algorithm::hat( const Eigen::Vector3d& vec )
{
    Eigen::Matrix3d skew;
    skew << 0.0, -vec.z(), vec.y(), vec.z(), 0.0, -vec.x(), -vec.y(), vec.x(), 0.0;
    return skew;
}

double algorithm::computeMedian( const Eigen::VectorXd& input )
{
    std::vector< double > vec( input.data(), input.data() + input.rows() * input.cols() );
    // return doubleVec[ middleSize ];
    const auto middleSize = vec.size() / 2;
    std::nth_element( vec.begin(), vec.begin() + middleSize, vec.end() );

    if ( vec.size() == 0 )
    {
        return std::numeric_limits< double >::quiet_NaN();
    }
    else if ( vec.size() % 2 != 0 )  // Odd
    {
        return vec[ middleSize ];
    }
    else  // Even
    {
        return ( vec[ middleSize - 1 ] + vec[ middleSize ] ) / 2.0;
    }
}

double algorithm::computeMedian( const Eigen::VectorXd& input, const uint32_t numValidPoints )
{
    std::vector< double > vec( input.data(), input.data() + input.rows() * input.cols() );

    const auto middleSize = numValidPoints / 2;
    std::nth_element( vec.begin(), vec.begin() + middleSize, vec.end() );

    if ( vec.size() == 0 )
    {
        return std::numeric_limits< double >::quiet_NaN();
    }
    else if ( vec.size() % 2 != 0 )  // Odd
    {
        return vec[ middleSize ];
    }
    else  // Even
    {
        return ( vec[ middleSize - 1 ] + vec[ middleSize ] ) / 2.0;
    }
}

double algorithm::computeMAD( const Eigen::VectorXd& input, const uint32_t numValidPoints )
{
    const std::size_t numObservations = input.rows();
    const double median               = computeMedian( input, numValidPoints );
    Eigen::VectorXd diffWithMedian( numObservations );
    for ( std::size_t i( 0 ); i < numObservations; i++ )
    {
        diffWithMedian( i ) = std::abs( input( i ) - median );
    }
    return computeMedian( diffWithMedian, numValidPoints );
}

double algorithm::computeSigma( const Eigen::VectorXd& input, const uint32_t numValidPoints, const double k )
{
    const double mad = computeMAD( input, numValidPoints );
    // Algorithm_Log( DEBUG ) << "MAD: " << mad;
    return k * mad;
}

float algorithm::bilinearInterpolation( const MapXRow& image, const double x, const double y )
{
    const int x1  = static_cast< int >( x );
    const int y1  = static_cast< int >( y );
    const int x2  = x1 + 1;
    const int y2  = y1 + 1;
    const float a = ( x2 - x ) * image( y1, x1 ) + ( x - x1 ) * image( y1, x2 );
    const float b = ( x2 - x ) * image( y2, x1 ) + ( x - x1 ) * image( y2, x2 );
    return ( ( y2 - y ) * a + ( y - y1 ) * b );
}

float algorithm::bilinearInterpolation( const MapXRowConst& image, const double x, const double y )
{
    const int x1  = static_cast< int >( x );
    const int y1  = static_cast< int >( y );
    const int x2  = x1 + 1;
    const int y2  = y1 + 1;
    const float a = ( x2 - x ) * image( y1, x1 ) + ( x - x1 ) * image( y1, x2 );
    const float b = ( x2 - x ) * image( y2, x1 ) + ( x - x1 ) * image( y2, x2 );
    return ( ( y2 - y ) * a + ( y - y1 ) * b );
}

double algorithm::computeNormalDistribution( const double mu, const double sigma, const double x )
{
    const double p = ( x - mu ) / sigma;
    return utils::constants::inv_sqrt_2_pi / sigma * std::exp( -0.5 * p * p );
}

// double computeMedianInplace( const Eigen::VectorXd& vec )
// {
//     const auto middleSize = vec.size() / 2;
//     // std::nth_element( vec.begin(), vec.begin() + middleSize, vec.end() );
//     auto beginIter = vec.data();
//     std::sort(beginIter, beginIter + middleSize);

//     if (vec.size() == 0)
//     {
//         return std::numeric_limits<double>::quiet_NaN();
//     }
//     else if(vec.size() % 2 != 0) // Odd
//     {
//         return vec(middleSize);
//     }
//     else //Even
//     {
//         return (vec(middleSize - 1) + vec(middleSize)) / 2.0;
//     }
// }
