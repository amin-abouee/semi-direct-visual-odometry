#include "algorithm.hpp"
#include "feature.hpp"

void Algorithm::pointsRefCamera( const Frame& refFrame, const Frame& curFrame, Eigen::MatrixXd& pointsRefCamera )
{
    const uint32_t featureSz = refFrame.numberObservation();
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
    const uint32_t featureSz = refFrame.numberObservation();
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
    const uint32_t featureSz = refFrame.numberObservation();
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
    const uint32_t featureSz = curFrame.numberObservation();
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

    Eigen::Vector2d project1 = refFrame.world2image( res.head( 2 ) );
    Eigen::Vector2d project2 = curFrame.world2image( res.head( 2 ) );
    std::cout << "Error in ref: " << ( project1 - refFeature ).norm()
              << ", Error in cur: " << ( project2 - curFeature ).norm() << std::endl;
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

    Eigen::Vector2d project1 = refFrame.world2image( point );
    Eigen::Vector2d project2 = curFrame.world2image( point );
    std::cout << "Error in ref: " << ( project1 - refFeature ).norm()
              << ", Error in cur: " << ( project2 - curFeature ).norm() << std::endl;
}

// 9.6.2 Extraction of cameras from the essential matrix, multi view geometry
// https://github.com/opencv/opencv/blob/a74fe2ec01d9218d06cb7675af633fc3f409a6a2/modules/calib3d/src/five-point.cpp
void Algorithm::decomposeEssentialMatrix( Eigen::Matrix3d& E,
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

bool Algorithm::checkCheirality()
{
    return true;
}