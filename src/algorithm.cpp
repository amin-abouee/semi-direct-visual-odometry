#include "algorithm.hpp"
#include "feature.hpp"

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

void Algorithm::pointsRefCamera( const Frame& refFrame, const Frame& curFrame, Eigen::MatrixXd& pointsRefCamera )
{
    const auto featureSz = refFrame.numberObservation();
    Eigen::Vector2d refFeature;
    Eigen::Vector2d curFeature;
    Eigen::Vector3d pointWorld;
    for ( std::size_t i( 0 ); i < featureSz; i++ )
    {
        refFeature = refFrame.m_frameFeatures[ i ]->m_feature;
        curFeature = curFrame.m_frameFeatures[ i ]->m_feature;
        triangulatePointDLT( refFrame, curFrame, refFeature, curFeature, pointWorld );
        pointsRefCamera.col( i ) = refFrame.world2camera( pointWorld );
    }
}

void Algorithm::pointsCurCamera( const Frame& refFrame, const Frame& curFrame, Eigen::MatrixXd& pointsCurCamera )
{
    const auto featureSz = refFrame.numberObservation();
    Eigen::Vector2d refFeature;
    Eigen::Vector2d curFeature;
    Eigen::Vector3d pointWorld;
    for ( std::size_t i( 0 ); i < featureSz; i++ )
    {
        refFeature = refFrame.m_frameFeatures[ i ]->m_feature;
        curFeature = curFrame.m_frameFeatures[ i ]->m_feature;
        triangulatePointDLT( refFrame, curFrame, refFeature, curFeature, pointWorld );
        pointsCurCamera.col( i ) = curFrame.world2camera( pointWorld );
    }
}

void Algorithm::normalizedDepthRefCamera( const Frame& refFrame,
                                          const Frame& curFrame,
                                          Eigen::VectorXd& depthRefCamera )
{
    const auto featureSz = refFrame.numberObservation();
    Eigen::MatrixXd pointsRefCamera( 3, featureSz );
    Algorithm::pointsRefCamera( refFrame, curFrame, pointsRefCamera );
    for ( std::size_t i( 0 ); i < featureSz; i++ )
    {
        depthRefCamera( i ) = pointsRefCamera.col( i ).norm();
    }
}

void Algorithm::normalizedDepthsCurCamera( const Frame& refFrame,
                                           const Frame& curFrame,
                                           Eigen::VectorXd& depthCurCamera )
{
    const auto featureSz = curFrame.numberObservation();
    Eigen::MatrixXd pointsCurCamera( 3, featureSz );
    Algorithm::pointsCurCamera( refFrame, curFrame, pointsCurCamera );
    for ( std::size_t i( 0 ); i < featureSz; i++ )
    {
        depthCurCamera( i ) = pointsCurCamera.col( i ).norm();
    }
}

void Algorithm::triangulatePointHomogenousDLT( const Frame& refFrame,
                                               const Frame& curFrame,
                                               const Eigen::Vector2d& refFeature,
                                               const Eigen::Vector2d& curFeature,
                                               Eigen::Vector3d& point )
{
    Eigen::MatrixXd A( 4, 4 );
    const Eigen::Matrix< double, 3, 4 > P1 = refFrame.m_camera->K() * refFrame.m_TransW2F.matrix3x4();
    const Eigen::Matrix< double, 3, 4 > P2 = curFrame.m_camera->K() * curFrame.m_TransW2F.matrix3x4();

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

void Algorithm::triangulatePointDLT( const Frame& refFrame,
                                     const Frame& curFrame,
                                     const Eigen::Vector2d& refFeature,
                                     const Eigen::Vector2d& curFeature,
                                     Eigen::Vector3d& point )
{
    Eigen::MatrixXd A( 4, 3 );
    const Eigen::Matrix< double, 3, 4 > P1 = refFrame.m_camera->K() * refFrame.m_TransW2F.matrix3x4();
    const Eigen::Matrix< double, 3, 4 > P2 = curFrame.m_camera->K() * curFrame.m_TransW2F.matrix3x4();

    A.row( 0 ) << P1( 0, 0 ) - refFeature.x() * P1( 2, 0 ), P1( 0, 1 ) - refFeature.x() * P1( 2, 1 ),
      P1( 0, 2 ) - refFeature.x() * P1( 2, 2 );
    A.row( 1 ) << P1( 1, 0 ) - refFeature.y() * P1( 2, 0 ), P1( 1, 1 ) - refFeature.y() * P1( 2, 1 ),
      P1( 1, 2 ) - refFeature.y() * P1( 2, 2 );
    A.row( 2 ) << P2( 0, 0 ) - curFeature.x() * P2( 2, 0 ), P2( 0, 1 ) - curFeature.x() * P2( 2, 1 ),
      P2( 0, 2 ) - curFeature.x() * P2( 2, 2 );
    A.row( 3 ) << P2( 1, 0 ) - curFeature.y() * P2( 2, 0 ), P2( 1, 1 ) - curFeature.y() * P2( 2, 1 ),
      P2( 1, 2 ) - curFeature.y() * P2( 2, 2 );

    Eigen::VectorXd p( 4 );
    p << refFeature.x() * P1( 2, 3 ) - P1( 0, 3 ), refFeature.y() * P1( 2, 3 ) - P1( 1, 3 ),
      curFeature.x() * P2( 2, 3 ) - P2( 0, 3 ), curFeature.y() * P2( 2, 3 ) - P2( 1, 3 );
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

// 9.6.2 Extraction of cameras from the essential matrix, multi view geometry
// https://github.com/opencv/opencv/blob/a74fe2ec01d9218d06cb7675af633fc3f409a6a2/modules/calib3d/src/five-point.cpp
void Algorithm::decomposeEssentialMatrix( const Eigen::Matrix3d& E,
                                          Eigen::Matrix3d& R1,
                                          Eigen::Matrix3d& R2,
                                          Eigen::Vector3d& t )
{
    Eigen::JacobiSVD< Eigen::Matrix3d > svd_E( E, Eigen::ComputeFullV | Eigen::ComputeFullU );
    Eigen::Matrix3d W;
    W << 0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0;
    R1 = svd_E.matrixU() * W * svd_E.matrixV().transpose();
    if ( R1.determinant() < 0 )
        R1 *= -1;
    R2 = svd_E.matrixU() * W.transpose() * svd_E.matrixV().transpose();
    if ( R2.determinant() < 0 )
        R2 *= -1;
    t = svd_E.matrixU().col( 2 );
}

void Algorithm::recoverPose(
  const Eigen::Matrix3d& E, const Frame& refFrame, Frame& curFrame, Eigen::Matrix3d& R, Eigen::Vector3d& t )
{
    Eigen::Matrix3d R1;
    Eigen::Matrix3d R2;
    // Eigen::Vector3d t;
    decomposeEssentialMatrix(E, R1, R2, t);

    Eigen::Vector2d topLeftCorner(1.0, 1.0);
    Eigen::Vector2d downRightCorner(refFrame.m_camera->width(), refFrame.m_camera->height());

    std::vector< Sophus::SE3d, Eigen::aligned_allocator< Sophus::SE3d > > poses;
    poses.reserve(4);
    Eigen::AngleAxisd temp( R1 );  // Re-orthogonality
    poses.emplace_back(Sophus::SE3d(temp.toRotationMatrix(), t ));
    poses.emplace_back(Sophus::SE3d(temp.toRotationMatrix(), -t ));
    temp = Eigen::AngleAxisd(R2);
    poses.emplace_back(Sophus::SE3d(temp.toRotationMatrix(), t ));
    poses.emplace_back(Sophus::SE3d(temp.toRotationMatrix(), -t ));

    for (std::size_t i(0); i<4; i++)
    {
        Eigen::Vector3d point1;
        Eigen::Vector3d point2;
        curFrame.m_TransW2F = refFrame.m_TransW2F * poses[i];
        triangulatePointDLT(refFrame, curFrame, topLeftCorner, topLeftCorner, point1);
        triangulatePointDLT(refFrame, curFrame, downRightCorner, downRightCorner, point2);
        Eigen::Vector3d refProject1 = refFrame.world2camera( point1 );
        Eigen::Vector3d curProject1 = curFrame.world2camera( point1 );
        Eigen::Vector3d refProject2 = refFrame.world2camera( point2 );
        Eigen::Vector3d curProject2 = curFrame.world2camera( point2 );
        std::cout << "output projct left corner, ref: " << refProject1.z() << ", cur: " << curProject1.z() << std::endl;
        std::cout << "output projct right corner, ref: " << refProject2.z() << ", cur: " << curProject2.z() << std::endl;
        if (refProject1.z() > 0 && refProject2.z() > 0 && curProject1.z() > 0 && curProject2.z() > 0)
        {
            R = poses[i].rotationMatrix();
            t = poses[i].translation();
            std::cout << "R: " << R << std::endl;
            std::cout << "t: " << t << std::endl;
            break;
        }
    }

}

bool Algorithm::checkCheirality()
{
    return true;
}

Eigen::Matrix3d Algorithm::hat( const Eigen::Vector3d& vec )
{
    // std::cout << "Vec: " << vec.transpose() << std::endl;
    Eigen::Matrix3d skew;
    skew << 0.0, -vec.z(), vec.y(), vec.z(), 0.0, -vec.x(), -vec.y(), vec.x(), 0.0;
    // std::cout << "skew: " << skew << std::endl;
    return skew;
}