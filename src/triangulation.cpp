#include "triangulation.hpp"

Triangulation::Triangulation()
{
}

void Triangulation::triangulatePointDLT( const Frame& refFrame,
                                    const Frame& curFrame,
                                    Eigen::Vector2d& refFeature,
                                    Eigen::Vector2d& curFeature,
                                    Eigen::Vector3d& point )
{
    Eigen::MatrixXd A (4, 4);
    const Eigen::Matrix<double, 3, 4> P1 = refFrame.m_camera->K() * refFrame.m_TransW2F.matrix3x4();
    // std::cout << "P1: " << P1 << std::endl;
    const Eigen::Matrix<double, 3, 4> P2 = curFrame.m_camera->K() * curFrame.m_TransW2F.matrix3x4();
    // std::cout << "P2: " << P2 << std::endl;

    A.row(0) = (refFeature.x() * P1.row(2)) - P1.row(0);
    A.row(1) = (refFeature.y() * P1.row(2)) - P1.row(1);
    A.row(2) = (curFeature.x() * P2.row(2)) - P2.row(0);
    A.row(3) = (curFeature.y() * P2.row(2)) - P2.row(1);

    // A.row(0) = refFeature.y() * P1.row(2) - P1.row(1);
    // A.row(1) = P1.row(0) - refFeature.x() * P1.row(2);
    // A.row(0) = curFeature.y() * P2.row(2) - P2.row(1);
    // A.row(1) = P2.row(0) - curFeature.x() * P2.row(2);

    // std::cout << "A: " << A << std::endl;


    A.row(0) /= A.row(0).norm(); 
    A.row(1) /= A.row(1).norm(); 
    A.row(2) /= A.row(2).norm(); 
    A.row(3) /= A.row(3).norm(); 

    // std::cout << "A norm: " << A << std::endl;
    // std::cout << "A size: " << A.rows() << " , " << A.cols() << std::endl;
    Eigen::JacobiSVD< Eigen::MatrixXd > svd_A( A.transpose() * A, Eigen::ComputeFullV );
    // std::cout << "Singular Values of AX=0: " << svd_A.singularValues().transpose() << std::endl;
    Eigen::VectorXd res = svd_A.matrixV().col(3);
    res /= res(3);
    std::cout << "res: " << res.transpose() << std::endl;
    if (res.z() < 0)
        res *= -1;
    std::cout << "x: " << res.x() << ", y: " << res.y() << ", z: " << res.z() << ", w: " << res.w() << std::endl;

    Eigen::Vector2d project1 = refFrame.world2image(res.head(2));
    Eigen::Vector2d project2 = curFrame.world2image(res.head(2));
    std::cout << "project 1: " << project1.transpose() << std::endl;
    std::cout << "project 2: " << project2.transpose() << std::endl;
    std::cout << "point in reference camera: " << refFrame.world2camera(res.head(2)).transpose() << std::endl;
    std::cout << "point in current camera: " << curFrame.world2camera(res.head(2)).transpose() << std::endl;
    // Eigen::VectorXd res2 = Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 4, 4>>(A).eigenvectors().col(3);
    // res2 /= res2(3);
    // std::cout << "res 2: " << res2 << std::endl;
}