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

// #include <spdlog/spdlog.h>
// #include "spdlog/sinks/stdout_color_sinks.h"
#include <nlohmann/json.hpp>
#include <easylogging++.h>
INITIALIZE_EASYLOGGINGPP


int main( int argc, char* argv[] )
{

#ifdef EIGEN_MALLOC_ALREADY_ALIGNED
    std::cout << "EIGEN_MALLOC_ALREADY_ALIGNED" << std::endl;
#endif

#ifdef EIGEN_FAST_MATH
    std::cout << "EIGEN_FAST_MATH" << std::endl;
#endif

#ifdef EIGEN_USE_MKL_ALL
    std::cout << "EIGEN_USE_MKL_ALL" << std::endl;
#endif

#ifdef EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    std::cout << "EIGEN_MAKE_ALIGNED_OPERATOR_NEW" << std::endl;
#endif

#ifdef EIGEN_USE_BLAS
    std::cout << "EIGEN_USE_BLAS" << std::endl;
#endif
    std::cout << "EIGEN_HAS_CXX11_CONTAINERS: " << EIGEN_HAS_CXX11_CONTAINERS << std::endl;
    std::cout << "EIGEN_MAX_CPP_VER: " << EIGEN_MAX_CPP_VER << std::endl;
    std::cout << "EIGEN_HAS_CXX11_MATH: " << EIGEN_HAS_CXX11_MATH << std::endl;

    el::Loggers::addFlag( el::LoggingFlag::MultiLoggerSupport );
    el::Loggers::addFlag( el::LoggingFlag::ColoredTerminalOutput );
    // el::Loggers::configureFromGlobal( logFileName.c_str() );
    // el::Loggers::reconfigureAllLoggers(conf);
    // el::Logger* systemLogger = el::Loggers::getLogger("system"); // Register new logger
    el::Loggers::getLogger( "Main" );  // Register new logger
    #define SYSTEM_LOG( LEVEL ) CLOG( LEVEL, "Main" )
    // std::cout << "Number of Threads: " << Eigen::nbThreads( ) << std::endl;
    // Eigen::setNbThreads(4);
    // std::cout << "Number of Threads: " << Eigen::nbThreads( ) << std::endl;

    // auto mainLogger = spdlog::stdout_color_mt( "main" );
    // mainLogger->set_level( spdlog::level::debug );
    // mainLogger->set_pattern( "[%Y-%m-%d %H:%M:%S] [%s:%#] [%n->%l] [thread:%t] %v" );
    // spdlog::set_pattern("[%H:%M:%S %z] [%n] [%^---%L---%$] [thread %t] %v");
    // mainLogger->info( "Info" );
    // mainLogger->debug( "Debug" );
    // mainLogger->warn( "Warn" );
    // mainLogger->error( "Error" );
    SYSTEM_LOG( INFO ) << "Scale";

    std::string configIOFile;
    if ( argc > 1 )
        configIOFile = argv[ 1 ];
    else
        configIOFile = "config/config.json";

    // Config::init(utils::findAbsoluteFilePath( configIOFile ));
    // Config* config = Config::getInstance();
    Config config( utils::findAbsoluteFilePath( configIOFile ) );
    System system( config );

    const cv::Mat refImg = cv::imread( utils::findAbsoluteFilePath( "input/0000000000.png" ), cv::IMREAD_GRAYSCALE );
    const cv::Mat curImg = cv::imread( utils::findAbsoluteFilePath( "input/0000000001.png" ), cv::IMREAD_GRAYSCALE );

    system.processFirstFrame( refImg );

    auto t1 = std::chrono::high_resolution_clock::now();
    system.processSecondFrame( curImg );
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "Elapsed time for matching: " << std::chrono::duration_cast< std::chrono::microseconds >( t2 - t1 ).count() << " micro sec"
              << std::endl;

    /*
        numObserves = refFrame.numberObservation();
        Eigen::VectorXd newCurDepths( numObserves );
        algorithm::depthCamera( curFrame, newCurDepths );
        medianDepth           = algorithm::computeMedian( newCurDepths );
        const double minDepth = newCurDepths.minCoeff();
        // std::cout << "Mean: " << medianDepth << " min: " << minDepth << std::endl;
        curFrame.setKeyframe();
    */

    for ( int i( 2 ); i < 10; i++ )
    {
        std::stringstream ss;
        ss << "input/000000000" << i << ".png";
        std::cout << "filename: " << ss.str() << std::endl;
        cv::Mat newImg = cv::imread( utils::findAbsoluteFilePath( ss.str() ), cv::IMREAD_GRAYSCALE );
        system.processNewFrame( newImg );
        cv::waitKey( 0 );
    }
    // system.reportSummaryFrames();
    // system.reportSummaryFeatures();
    return 0;
}
