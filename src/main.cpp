// #define EIGEN_DEFAULT_DENSE_INDEX_TYPE long
// #define EIGEN_DEFAULT_IO_FORMAT Eigen::IOFormat( 10, Eigen::DontAlignCols, ", ", " , ", "[", "]", "[", "]" )

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
// #include <opencv2/core/eigen.hpp>

#include <Eigen/Core>

#include "algorithm.hpp"
#include "system.hpp"
#include "utils.hpp"

#include <nlohmann/json.hpp>
#include "easylogging++.h"

INITIALIZE_EASYLOGGINGPP
#define Main_Log( LEVEL ) CLOG( LEVEL, "Main" )

void checkEigenFlags()
{
#ifdef EIGEN_MALLOC_ALREADY_ALIGNED
    Main_Log( DEBUG ) << "EIGEN_MALLOC_ALREADY_ALIGNED";
#endif

#ifdef EIGEN_FAST_MATH
    Main_Log( DEBUG ) << "EIGEN_FAST_MATH";
#endif

#ifdef EIGEN_USE_MKL_ALL
    Main_Log( DEBUG ) << "EIGEN_USE_MKL_ALL";
#endif

#ifdef EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Main_Log( DEBUG ) << "EIGEN_MAKE_ALIGNED_OPERATOR_NEW";
#endif

#ifdef EIGEN_USE_BLAS
    Main_Log( DEBUG ) << "EIGEN_USE_BLAS";
#endif
    Main_Log( DEBUG ) << "EIGEN_HAS_CXX11_CONTAINERS: " << EIGEN_HAS_CXX11_CONTAINERS;
    Main_Log( DEBUG ) << "EIGEN_MAX_CPP_VER: " << EIGEN_MAX_CPP_VER;
    Main_Log( DEBUG ) << "EIGEN_HAS_CXX11_MATH: " << EIGEN_HAS_CXX11_MATH;
}

void configLogger( const std::string& logFilePath )
{
    el::Loggers::configureFromGlobal( utils::findAbsoluteFilePath( logFilePath ).c_str() );
    el::Loggers::addFlag( el::LoggingFlag::MultiLoggerSupport );
    el::Loggers::addFlag( el::LoggingFlag::ColoredTerminalOutput );
}

int main( int argc, char* argv[] )
{
    // Main_Log(DEBUG) << "Number of Threads: " << Eigen::nbThreads( );
    // Eigen::setNbThreads(4);
    // Main_Log(DEBUG) << "Number of Threads: " << Eigen::nbThreads( );
    std::string configIOFile;
    if ( argc > 1 )
        configIOFile = argv[ 1 ];
    else
        configIOFile = "config/config.json";

    // Config::init(utils::findAbsoluteFilePath( configIOFile ));
    // Config* config = Config::getInstance();
    Config config( utils::findAbsoluteFilePath( configIOFile ) );
    System system( config );
    configLogger( config.m_logFilePath );

    // const cv::Mat refImg = cv::imread( utils::findAbsoluteFilePath( "input/0000000000.png" ), cv::IMREAD_GRAYSCALE );
    // const cv::Mat curImg = cv::imread( utils::findAbsoluteFilePath( "input/0000000001.png" ), cv::IMREAD_GRAYSCALE );

    // system.processFirstFrame( refImg );

    // system.processSecondFrame( curImg );

    /*
        numObserves = refFrame.numberObservation();
        Eigen::VectorXd newCurDepths( numObserves );
        algorithm::depthCamera( curFrame, newCurDepths );
        medianDepth           = algorithm::computeMedian( newCurDepths );
        const double minDepth = newCurDepths.minCoeff();
        // std::cout << "Mean: " << medianDepth << " min: " << minDepth << std::endl;
        curFrame.setKeyframe();
    */

    for ( int i( 0 ); i < 10; i++ )
    {
        std::stringstream ss;
        ss << "input/000000000" << i << ".png";
        Main_Log( INFO ) << "filename: " << ss.str();
        cv::Mat newImg = cv::imread( utils::findAbsoluteFilePath( ss.str() ), cv::IMREAD_GRAYSCALE );
        auto t1        = std::chrono::high_resolution_clock::now();
        system.addImage( newImg, i );
        auto t2 = std::chrono::high_resolution_clock::now();
        Main_Log( DEBUG ) << "Elapsed time for matching: " << std::chrono::duration_cast< std::chrono::milliseconds >( t2 - t1 ).count()
                          << " milli sec";
        cv::waitKey( 0 );
    }
    // system.reportSummaryFrames();
    // system.reportSummaryFeatures();
    return 0;
}
