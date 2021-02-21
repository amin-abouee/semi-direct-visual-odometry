#include "algorithm.hpp"
#include "feature.hpp"
#include "feature_alignment.hpp"
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

bool algorithm::computeOpticalFlowSparse( std::shared_ptr< Frame >& refFrame,
                                          std::shared_ptr< Frame >& curFrame,
                                          const uint32_t patchSize,
                                          const double disparityThreshold )
{
    // TIMED_FUNC( timerOpticalFlow );

    const cv::Mat& refImg         = refFrame->m_imagePyramid.getBaseImage();
    const cv::Mat& curImg         = curFrame->m_imagePyramid.getBaseImage();
    const uint64_t refObservation = refFrame->numberObservation();
    Algorithm_Log( DEBUG ) << "Ref Observation before OF: " << refObservation;

    std::vector< cv::Point2f > refPoints;
    refPoints.reserve( refObservation );
    std::vector< cv::Point2f > curPoints;
    curPoints.reserve( refObservation );
    std::vector< uchar > status;
    status.reserve( refObservation );
    std::vector< float > errors;
    errors.reserve( refObservation );
    const int maxIteration    = 30;
    const double epsilonError = 1e-4;

    for ( const auto& features : refFrame->m_features )
    {
        refPoints.emplace_back(
          cv::Point2f( static_cast< float >( features->m_pixelPosition.x() ), static_cast< float >( features->m_pixelPosition.y() ) ) );
        curPoints.emplace_back(
          cv::Point2f( static_cast< float >( features->m_pixelPosition.x() ), static_cast< float >( features->m_pixelPosition.y() ) ) );
    }

    cv::TermCriteria termcrit( cv::TermCriteria::COUNT + cv::TermCriteria::EPS, maxIteration, epsilonError );
    cv::calcOpticalFlowPyrLK( refImg, curImg, refPoints, curPoints, status, errors, cv::Size( patchSize, patchSize ), 3, termcrit,
                              cv::OPTFLOW_USE_INITIAL_FLOW );

    std::vector< double > disparity;
    disparity.reserve( refObservation );
    for ( std::size_t i( 0 ); i < curPoints.size(); i++ )
    {
        if ( status[ i ] == true )
        {
            std::shared_ptr< Feature > newFeature =
              std::make_shared< Feature >( curFrame, Eigen::Vector2d( curPoints[ i ].x, curPoints[ i ].y ), 0.0, 0.0, 0 );
            curFrame->addFeature( newFeature );

            disparity.push_back( Eigen::Vector2d( refPoints[ i ].x - curPoints[ i ].x, refPoints[ i ].y - curPoints[ i ].y ).norm() );
        }
    }

    Eigen::Map< Eigen::Matrix< double, 1, Eigen::Dynamic > > mapDisparity( disparity.data(), disparity.size() );
    double medianDisparity = algorithm::computeMedian( mapDisparity );
    Algorithm_Log( DEBUG ) << "Disparity: " << medianDisparity;
    if ( medianDisparity < disparityThreshold )
    {
        return false;
    }

    uint32_t cnt = 0;
    /// if status[i] == true, it have to return false because we dont want to remove it from our container
    auto isNotValid = [ &cnt, &status ]( const auto& feature ) {
        if ( feature != nullptr && status[ cnt++ ] == 1 )
        {
            return false;
        }
        else
        {
            return true;
        }
    };

    // https://en.wikipedia.org/wiki/Erase%E2%80%93remove_idiom
    // removed non-tracked points from reference frame
    refFrame->m_features.erase( std::remove_if( refFrame->m_features.begin(), refFrame->m_features.end(), isNotValid ),
                                refFrame->m_features.end() );

    Algorithm_Log( DEBUG ) << "After OF Observation refFrame: " << refFrame->numberObservation();
    Algorithm_Log( DEBUG ) << "After OF Observation curFrame: " << curFrame->numberObservation();
    return true;
}

bool algorithm::computeEssentialMatrix( std::shared_ptr< Frame >& refFrame,
                                        std::shared_ptr< Frame >& curFrame,
                                        const double reproError,
                                        const uint32_t thresholdCorrespondingPoints,
                                        Eigen::Matrix3d& E )
{
    // TIMED_FUNC( timerComputeEssentialMatrix );

    std::vector< cv::Point2f > refPoints;
    std::vector< cv::Point2f > curPoints;
    std::vector< uchar > status;
    const std::size_t featureSize = refFrame->numberObservation();

    for ( std::size_t i( 0 ); i < featureSize; i++ )
    {
        refPoints.emplace_back( cv::Point2f( static_cast< float >( refFrame->m_features[ i ]->m_pixelPosition.x() ),
                                             static_cast< float >( refFrame->m_features[ i ]->m_pixelPosition.y() ) ) );
        curPoints.emplace_back( cv::Point2f( static_cast< float >( curFrame->m_features[ i ]->m_pixelPosition.x() ),
                                             static_cast< float >( curFrame->m_features[ i ]->m_pixelPosition.y() ) ) );
    }

    cv::Mat E_cv = cv::findEssentialMat( refPoints, curPoints, refFrame->m_camera->K_cv(), cv::RANSAC, 0.999, reproError, status );

    double* essential = E_cv.ptr< double >( 0 );
    E << essential[ 0 ], essential[ 1 ], essential[ 2 ], essential[ 3 ], essential[ 4 ], essential[ 5 ], essential[ 6 ], essential[ 7 ],
      essential[ 8 ];

    Algorithm_Log( DEBUG ) << "E: " << E.format( utils::eigenFormat() );
    

    // https://stackoverflow.com/a/23123481/1804533
    uint32_t cnt = 0;
    /// if status[i] == true, it have to return false because we dont want to remove it from our container
    auto isNotValidRefFrame = [ &cnt, &status, &refFrame ]( const auto& feature ) {
        if ( feature->m_frame == refFrame )
            return status[ cnt++ ] ? false : true;
        else
            return false;
    };

    // https://en.wikipedia.org/wiki/Erase%E2%80%93remove_idiom
    auto refResult = std::remove_if( refFrame->m_features.begin(), refFrame->m_features.end(), isNotValidRefFrame );
    refFrame->m_features.erase( refResult, refFrame->m_features.end() );

    cnt                     = 0;
    auto isNotValidCurFrame = [ &cnt, &status, &curFrame ]( const auto& feature ) {
        if ( feature->m_frame == curFrame )
            return status[ cnt++ ] ? false : true;
        else
            return false;
    };
    auto curResult = std::remove_if( curFrame->m_features.begin(), curFrame->m_features.end(), isNotValidCurFrame );
    curFrame->m_features.erase( curResult, curFrame->m_features.end() );

    Algorithm_Log( DEBUG ) << "After ES Observation refFrame: " << refFrame->numberObservation();
    Algorithm_Log( DEBUG ) << "After ES Observation curFrame: " << curFrame->numberObservation();

    if ( refFrame->numberObservation() > thresholdCorrespondingPoints && curFrame->numberObservation() > thresholdCorrespondingPoints )
    {
        return true;
    }
    return false;
}

void algorithm::sampsonCorrection( std::shared_ptr< Frame >& refFrame, std::shared_ptr< Frame >& curFrame, const Eigen::Matrix3d& F )
{
    TIMED_FUNC( timerSampsonCorrection );

    // Used variables
    double sampson_factor    = 0.0;
    double sampson_error     = 0.0;
    double upper_factor_part = 0.0;
    double lower_factor_part = 0.0;

    double t0 = 0.0;
    double t1 = 0.0;
    double t2 = 0.0;
    double t3 = 0.0;

    Eigen::Vector2d x0;
    Eigen::Vector2d x1;

    Eigen::Vector2d x0_corr;
    Eigen::Vector2d x1_corr;

    const std::size_t featureSize = refFrame->numberObservation();

    for ( std::size_t id( 0 ); id < featureSize; id++ )
    {
        /* Get points from image Im0 and Im1 */
        x0 = refFrame->m_features[ id ]->m_pixelPosition;

        x1 = curFrame->m_features[ id ]->m_pixelPosition;

        /* See Multi-view geometry, p.315, Section 12.4 */
        /* Compute x1^T * F * x0 */
        upper_factor_part = ( x0.x() * ( ( x1.x() * F( 0, 0 ) ) + ( x1.y() * F( 1, 0 ) ) + F( 2, 0 ) ) ) +
                            ( x0.y() * ( ( x1.x() * F( 0, 1 ) ) + ( x1.y() * F( 1, 1 ) ) + F( 2, 1 ) ) ) + ( x1.x() * F( 0, 2 ) ) +
                            ( x1.y() * F( 1, 2 ) ) + F( 2, 2 );

        /* Parts of lower term */
        t0 = ( ( F( 0, 0 ) * x0.x() ) + ( F( 0, 1 ) * x0.y() ) + F( 0, 2 ) );

        t1 = ( ( F( 1, 0 ) * x0.x() ) + ( F( 1, 1 ) * x0.y() ) + F( 1, 2 ) );

        t2 = ( ( F( 0, 0 ) * x1.x() ) + ( F( 1, 0 ) * x1.y() ) + F( 2, 0 ) );

        t3 = ( ( F( 0, 1 ) * x1.x() ) + ( F( 1, 1 ) * x1.y() ) + F( 2, 1 ) );

        lower_factor_part = ( ( t0 * t0 ) + ( t1 * t1 ) + ( t2 * t2 ) + ( t3 * t3 ) );

        /* Compute Sampson factor */
        sampson_factor = upper_factor_part / lower_factor_part;

        /* Compute Sampson error (see Multi-view geometry, p.287, Eq. 11.9) */
        sampson_error += (sampson_factor)*upper_factor_part;

        /* Corrected point coordinates */
        x0_corr.x() = x0.x() - ( sampson_factor * t2 );
        x0_corr.y() = x0.y() - ( sampson_factor * t3 );
        x1_corr.x() = x1.x() - ( sampson_factor * t0 );
        x1_corr.y() = x1.y() - ( sampson_factor * t1 );

        /* Assign the corrected values to the reference and current frame */
        refFrame->m_features[ id ]->m_pixelPosition = x0_corr;
        curFrame->m_features[ id ]->m_pixelPosition = x1_corr;
    }
    Algorithm_Log( DEBUG ) << "Total sampson error: " << sampson_error;
}

// 9.6.2 Extraction of cameras from the essential matrix, multi view geometry
// https://github.com/opencv/opencv/blob/a74fe2ec01d9218d06cb7675af633fc3f409a6a2/modules/calib3d/src/five-point.cpp
void algorithm::decomposeEssentialMatrix( const Eigen::Matrix3d& E, Eigen::Matrix3d& R1, Eigen::Matrix3d& R2, Eigen::Vector3d& t )
{
    Eigen::JacobiSVD< Eigen::Matrix3d > svd_E( E, Eigen::ComputeFullV | Eigen::ComputeFullU );
    // TODO: check why this W
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
}

bool algorithm::recoverPose( const Eigen::Matrix3d& E,
                             const std::shared_ptr< Frame >& refFrame,
                             std::shared_ptr< Frame >& curFrame,
                             Eigen::Matrix3d& R,
                             Eigen::Vector3d& t )
{
    // TIMED_FUNC( timerRecoverPose );

    Eigen::Matrix3d R1;
    Eigen::Matrix3d R2;
    Eigen::Vector3d tm;
    decomposeEssentialMatrix( E, R1, R2, tm );

    Algorithm_Log( DEBUG ) << "R1: " << R1.format( utils::eigenFormat() );
    Algorithm_Log( DEBUG ) << "R2: " << R2.format( utils::eigenFormat() );
    Algorithm_Log( DEBUG ) << "tm: " << tm.format( utils::eigenFormat() );


    std::vector< Sophus::SE3d, Eigen::aligned_allocator< Sophus::SE3d > > poses;
    poses.reserve( 4 );
    Eigen::AngleAxisd temp( R1 );  // Re-orthogonality
    poses.emplace_back( Sophus::SE3d( temp.toRotationMatrix(), tm ) );
    poses.emplace_back( Sophus::SE3d( temp.toRotationMatrix(), -tm ) );
    temp = Eigen::AngleAxisd( R2 );
    poses.emplace_back( Sophus::SE3d( temp.toRotationMatrix(), tm ) );
    poses.emplace_back( Sophus::SE3d( temp.toRotationMatrix(), -tm ) );

    int32_t winnerIndex      = -1;
    uint32_t numberProjected = 0;

    for ( int32_t i( 0 ); i < 4; i++ )
    {
        Eigen::Vector3d point1;
        Eigen::Vector3d point2;

        // T{K}_{W} = T{K}_{K-1} * T{K-1}_{W}
        curFrame->m_absPose = poses[ i ] * refFrame->m_absPose;

        uint32_t numObserves = (uint32_t)curFrame->numberObservation();
        Eigen::MatrixXd pointsWorld( 3, numObserves );
        Eigen::MatrixXd pointsCurCamera( 3, numObserves );
        Eigen::MatrixXd pointsRefCamera( 3, numObserves );

        // compute the 3D points
        algorithm::triangulate3DWorldPoints( refFrame, curFrame, pointsWorld );
        algorithm::transferPointsWorldToCam( refFrame, pointsWorld, pointsCurCamera );
        algorithm::transferPointsWorldToCam( curFrame, pointsWorld, pointsRefCamera );
        uint32_t cntProjected = 0;
        for ( uint32_t j( 0 ); j < numObserves; j++ )
        {
            if ( pointsCurCamera.col( j ).z() > 0 && pointsRefCamera.col( j ).z() > 0 )
            {
                cntProjected++;
            }
        }

        Algorithm_Log (DEBUG) << poses[i].params().transpose() << ", num: " << cntProjected;
        if ( cntProjected > numberProjected )
        {
            numberProjected = cntProjected;
            winnerIndex     = i;
        }
    }

    if ( winnerIndex > -1 )
    {
        R                        = poses[ winnerIndex ].rotationMatrix();
        t                        = poses[ winnerIndex ].translation();
        curFrame->m_absPose = poses[ winnerIndex ];
        return true;
    }
    return false;
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
    // |------c-------u     ====>       |---c'--------->                            ==> affineWarp = [duDiff, dvDiff]
    // |-------------->                 |-------------->
    // |-------------->                 |-------------->    ==> dvDiff = v' - c'
    // |------v------->                 |--------v'---->

    const Eigen::Vector3d centerRefCamera = refFrame->image2camera( feature->m_pixelPosition, depth );
    const Eigen::Vector3d duRefCamera = refFrame->image2camera( feature->m_pixelPosition + Eigen::Vector2d( halfPatchSize, 0.0 ), depth );
    const Eigen::Vector3d dvRefCamera = refFrame->image2camera( feature->m_pixelPosition + Eigen::Vector2d( 0.0, halfPatchSize ), depth );

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

    uint32_t idx                = 0;
    const Eigen::Vector2d bound = affineWarp * Eigen::Vector2d( halfPatchSize, halfPatchSize );
    const double maxBoundary    = std::ceil( std::max( std::fabs( bound.x() ), std::fabs( bound.y() ) ) ) + 2;
    if ( frame->m_camera->isInFrame( location, maxBoundary ) )
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

bool algorithm::matchEpipolarConstraint( const std::shared_ptr< Frame >& refFrame,
                                         const std::shared_ptr< Frame >& curFrame,
                                         std::shared_ptr< Feature >& refFeature,
                                         const uint32_t patchSize,
                                         const double initialDepth,
                                         const double minDepth,
                                         const double maxDepth,
                                         double& estimatedDepth )
{
    std::shared_ptr< FeatureAlignment > alignment = std::make_shared< FeatureAlignment >( patchSize, 0, 3 );
    // const uint32_t patchSize     = 7;
    const uint32_t halfPatchSize = patchSize / 2;
    const uint32_t patchArea     = patchSize * patchSize;

    const Sophus::SE3d relativePose = algorithm::computeRelativePose( refFrame, curFrame );
    const uint32_t thresholdZSSD    = patchArea * 128;
    Algorithm_Log( DEBUG ) << "depth: " << initialDepth << ", min: " << minDepth << ", max: " << maxDepth
                           << ", Thr ZZSD: " << thresholdZSSD;
    Eigen::Vector2d locationCenter =
      curFrame->camera2image( relativePose * refFrame->image2camera( refFeature->m_pixelPosition, initialDepth ) );
    Eigen::Vector2d locationMin = curFrame->camera2image( relativePose * refFrame->image2camera( refFeature->m_pixelPosition, minDepth ) );
    Eigen::Vector2d locationMax = curFrame->camera2image( relativePose * refFrame->image2camera( refFeature->m_pixelPosition, maxDepth ) );

    {
        locationCenter.x() = locationCenter.x() >= 0 ? locationCenter.x() : 0.0;
        locationCenter.x() = locationCenter.x() < curFrame->m_camera->width() ? locationCenter.x() : curFrame->m_camera->width() - 1;
        locationCenter.y() = locationCenter.y() >= 0 ? locationCenter.y() : 0.0;
        locationCenter.y() = locationCenter.y() < curFrame->m_camera->height() ? locationCenter.y() : curFrame->m_camera->height() - 1;

        locationMin.x() = locationMin.x() >= 0 ? locationMin.x() : 0.0;
        locationMin.x() = locationMin.x() < curFrame->m_camera->width() ? locationMin.x() : curFrame->m_camera->width() - 1;
        locationMin.y() = locationMin.y() >= 0 ? locationMin.y() : 0.0;
        locationMin.y() = locationMin.y() < curFrame->m_camera->height() ? locationMin.y() : curFrame->m_camera->height() - 1;

        locationMax.x() = locationMax.x() >= 0 ? locationMax.x() : 0.0;
        locationMax.x() = locationMax.x() < curFrame->m_camera->width() ? locationMax.x() : curFrame->m_camera->width() - 1;
        locationMax.y() = locationMax.y() >= 0 ? locationMax.y() : 0.0;
        locationMax.y() = locationMax.y() < curFrame->m_camera->height() ? locationMax.y() : curFrame->m_camera->height() - 1;
    }

    Algorithm_Log( DEBUG ) << "Center: " << locationCenter.transpose() << ", min: " << locationMin.transpose()
                           << ", max: " << locationMax.transpose();

    // FIXME: check with the original code. They don't project to the camera. Just project2d (x/z, y/z)
    const Eigen::Vector2d epipolarDirection = locationMax - locationMin;
    Algorithm_Log( DEBUG ) << "epipolarDirection: " << epipolarDirection.transpose();

    Eigen::Matrix2d affineWarp;
    algorithm::getAffineWarp( refFrame, curFrame, refFeature, relativePose, patchSize, initialDepth, affineWarp );
    double normEpipolar = epipolarDirection.norm();
    Algorithm_Log( DEBUG ) << "affine matrix: " << affineWarp.format( utils::eigenFormat() );

    Eigen::Matrix< uint8_t, Eigen::Dynamic, 1 > refPatchIntensities( patchArea );
    refPatchIntensities.setZero();
    algorithm::applyAffineWarp( refFrame, refFeature->m_pixelPosition, halfPatchSize, Eigen::Matrix2d::Identity(), halfPatchSize + 2,
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
    const double boundary         = std::ceil( std::max( borders.x(), borders.y() ) ) + 1;

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
    // Algorithm_Log( INFO ) << "Location in ref frame: " << refFeature->m_pixelPosition.transpose();
    // Algorithm_Log( INFO ) << "Ref patch intensities: " << refPatchIntensities.cast<int32_t>().transpose();

    // traverse over the epipolar line
    for ( uint32_t i( 0 ); i < pixelStep; i++ )
    {
        const Eigen::Vector2d location = locationMin + i * step2D;
        Algorithm_Log( DEBUG ) << "loc for point: " << location.transpose();
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
        // Algorithm_Log( DEBUG ) << "first best location: " << bestLocation.transpose();
        // double error = alignment->align( refFeature, curFrame, bestLocation );
        // Algorithm_Log( DEBUG ) << "error: " << error << ", best location: " << bestLocation.transpose();

        // {
        //     algorithm::applyAffineWarp( curFrame, bestLocation, halfPatchSize, affineWarp, boundary, curPatchIntensities );
        //     double zssd = computeScore( refPatchIntensities, curPatchIntensities );
        //     Algorithm_Log( DEBUG ) << "new zssd score: " << zssd;
        // }

        const Eigen::Vector3d bearingCurCamera = curFrame->m_camera->inverseProject2d( bestLocation );
        Algorithm_Log( DEBUG ) << "ref location: " << refFeature->m_pixelPosition.transpose()
                               << ", cur location: " << bestLocation.transpose() << ", pre location: " << locationCenter.transpose();

        bool resTriangulation = depthFromTriangulation( relativePose, refFeature->m_bearingVec, bearingCurCamera, estimatedDepth );
        if ( resTriangulation == true )
        {
            Algorithm_Log( DEBUG ) << "pre depth: " << initialDepth << ", depth updated: " << estimatedDepth;
            return true;
        }
    }

    return false;
}

void algorithm::triangulate3DWorldPoints( const std::shared_ptr< Frame >& refFrame,
                                          const std::shared_ptr< Frame >& curFrame,
                                          Eigen::MatrixXd& pointsWorld )
{
    const auto featureSz = refFrame->numberObservation();
    Eigen::Vector3d pointWorld;
    for ( std::size_t i( 0 ); i < featureSz; i++ )
    {
        const Eigen::Vector2d refFeature = refFrame->m_features[ i ]->m_pixelPosition;
        const Eigen::Vector2d curFeature = curFrame->m_features[ i ]->m_pixelPosition;
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
        normalizedDepthCamera( i ) = frame->world2camera( frame->m_features[ i ]->m_point->m_position ).norm();
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
        depthCamera( i ) = frame->world2camera( frame->m_features[ i ]->m_point->m_position ).z();
    }
}

void algorithm::triangulatePointHomogenousDLT( const std::shared_ptr< Frame >& refFrame,
                                               const std::shared_ptr< Frame >& curFrame,
                                               const Eigen::Vector2d& refFeature,
                                               const Eigen::Vector2d& curFeature,
                                               Eigen::Vector3d& point )
{
    Eigen::MatrixXd A( 4, 4 );
    const Eigen::Matrix< double, 3, 4 > P1 = refFrame->m_camera->K() * refFrame->m_absPose.matrix3x4();
    const Eigen::Matrix< double, 3, 4 > P2 = curFrame->m_camera->K() * curFrame->m_absPose.matrix3x4();

    A.row( 0 ) = ( refFeature.x() * P1.row( 2 ) ) - P1.row( 0 );
    A.row( 1 ) = ( refFeature.y() * P1.row( 2 ) ) - P1.row( 1 );
    A.row( 2 ) = ( curFeature.x() * P2.row( 2 ) ) - P2.row( 0 );
    A.row( 3 ) = ( curFeature.y() * P2.row( 2 ) ) - P2.row( 1 );

    A.row( 0 ) /= A.row( 0 ).norm();
    A.row( 1 ) /= A.row( 1 ).norm();
    A.row( 2 ) /= A.row( 2 ).norm();
    A.row( 3 ) /= A.row( 3 ).norm();

    Eigen::JacobiSVD< Eigen::MatrixXd > svd_A( A.transpose() * A, Eigen::ComputeFullV );
    Eigen::VectorXd res = svd_A.matrixV().col( 3 );
    res /= res.w();

    point = res.head( 2 );
}

void algorithm::triangulatePointDLT( const std::shared_ptr< Frame >& refFrame,
                                     const std::shared_ptr< Frame >& curFrame,
                                     const Eigen::Vector2d& refFeature,
                                     const Eigen::Vector2d& curFeature,
                                     Eigen::Vector3d& point )
{
    Eigen::MatrixXd A( 4, 3 );
    const Eigen::Matrix< double, 3, 4 > P1 = refFrame->m_camera->K() * refFrame->m_absPose.matrix3x4();
    const Eigen::Matrix< double, 3, 4 > P2 = curFrame->m_camera->K() * curFrame->m_absPose.matrix3x4();
    // Algorithm_Log(DEBUG) << "pose reference: " << curFrame->m_absPose.params().transpose();

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
        Algorithm_Log( DEBUG ) << "det: " << AtA.determinant();
        return false;
    }
    const Eigen::Vector2d depths = -AtA.inverse() * A.transpose() * relativePose.translation();
    Algorithm_Log( DEBUG ) << "depths: " << depths.transpose();
    depth = std::fabs( depths.x() );
    return true;
}

Sophus::SE3d algorithm::computeRelativePose( const std::shared_ptr< Frame >& refFrame, const std::shared_ptr< Frame >& curFrame )
{
    // T{K}_{K-1} = T{K}_{W}T * T{W}_{K-1} = T{K}_{W} * T{K-1}_{W}^{-1}
    return curFrame->m_absPose * refFrame->m_absPose.inverse();
}

double algorithm::computeStructureError( const std::shared_ptr< Point >& point )
{
    double errors = 0;
    for ( const auto& feature : point->m_features )
    {
        const auto& frame                  = feature->m_frame;
        const Eigen::Vector2d projectPoint = frame->world2image( point->m_position );
        errors += ( feature->m_pixelPosition - projectPoint ).norm();
    }
    return errors;
}

double algorithm::computeStructureError( const std::shared_ptr< Frame >& frame )
{
    double errors = 0;
    for ( const auto& feature : frame->m_features )
    {
        if (feature->m_point != nullptr)
        {
            const Eigen::Vector2d projectPoint = frame->world2image( feature->m_point->m_position );
            errors += ( feature->m_pixelPosition - projectPoint ).norm();
        }
    }
    return errors;
}

double algorithm::computePatchError( const std::shared_ptr< Feature >& refFeature,
                                     const std::shared_ptr< Frame >& curFrame,
                                     const Eigen::Vector2d& curPosition,
                                     const int32_t patchSize )
{
    const int32_t halfPatchSize = patchSize / 2;
    const int32_t border        = halfPatchSize + 2;

    const std::shared_ptr< Frame >& refFrame = refFeature->m_frame;
    const cv::Mat& refImage                  = refFrame->m_imagePyramid.getGradientAtLevel( 0 );
    const algorithm::MapXRowConst refImageEigen( refImage.ptr< uint8_t >(), refImage.rows, refImage.cols );

    const cv::Mat& curImage = curFrame->m_imagePyramid.getGradientAtLevel( 0 );
    const algorithm::MapXRowConst curImageEigen( curImage.ptr< uint8_t >(), curImage.rows, curImage.cols );

    const Eigen::Vector2d refPosition = refFeature->m_pixelPosition;

    if ( curFrame->m_camera->isInFrame( curPosition, border ) == false )
    {
        return -1.0;
    }

    double totalError = 0.0;
    uint32_t cnt = 0;
    const int32_t beginIdx = -halfPatchSize;
    const int32_t endIdx   = halfPatchSize;
    for ( int32_t y{ beginIdx }; y <= endIdx; y++ )
    {
        for ( int32_t x{ beginIdx }; x <= endIdx; x++, cnt++ )
        {
            const double refRowIdx       = refPosition.y() + y;
            const double refColIdx       = refPosition.x() + x;
            const double refPixelValue = algorithm::bilinearInterpolationDouble( refImageEigen, refColIdx, refRowIdx );

            const double curRowIdx       = curPosition.y() + y;
            const double curColIdx       = curPosition.x() + x;
            const double curPixelValue = algorithm::bilinearInterpolationDouble( curImageEigen, curColIdx, curRowIdx );


            totalError += std::abs( curPixelValue - refPixelValue);
        }
    }

    return totalError/cnt;
}

uint32_t algorithm::computeNumberProjectedPoints( const std::shared_ptr< Frame >& curFrame )
{
    const auto& lastKFrame          = curFrame->m_lastKeyframe;
    const Sophus::SE3d relativePose = computeRelativePose( lastKFrame, curFrame );
    const Eigen::Vector3d C         = curFrame->cameraInWorld();
    uint32_t cntProjected           = 0;
    for ( const auto& feature : curFrame->m_features )
    {
        if ( feature->m_point != nullptr )
        {
            const double depthNorm = ( feature->m_point->m_position - C ).norm();
            const Eigen::Vector3d refPoint( feature->m_bearingVec * depthNorm );
            const Eigen::Vector3d curPoint( relativePose * refPoint );
            const Eigen::Vector2d curFeature( curFrame->camera2image( curPoint ) );
            if ( curFrame->m_camera->isInFrame( curFeature, 5.0 ) )
            {
                cntProjected++;
            }
        }
    }
    return cntProjected;
}

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

double algorithm::bilinearInterpolationDouble( const MapXRowConst& image, const double x, const double y )
{
    const int32_t x1 = static_cast< int32_t >( x );
    const int32_t y1 = static_cast< int32_t >( y );
    const int32_t x2 = x1 + 1;
    const int32_t y2 = y1 + 1;
    const double a   = ( x2 - x ) * image( y1, x1 ) + ( x - x1 ) * image( y1, x2 );
    const double b   = ( x2 - x ) * image( y2, x1 ) + ( x - x1 ) * image( y2, x2 );
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
