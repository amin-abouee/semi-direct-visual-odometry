#include "algorithm.hpp"
#include "system.hpp"
#include "utils.hpp"

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

#include <Eigen/Core>

#include <easylogging++.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <thread>

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
    // el::Helpers::setThreadName("main-thread");
}

int main( int argc, char* argv[] )
{
    TIMED_FUNC (timerMain);
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
    std::shared_ptr< Config > config = std::make_shared< Config >( utils::findAbsoluteFilePath( configIOFile ) );
    configLogger( config->m_logFilePath );
    System system( config );
    Main_Log( DEBUG ) << "Main thread id: " << std::this_thread::get_id();

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

    for ( int i( 0 ); i < 4; i++ )
    {
        std::stringstream ss;
        ss << "input/000000000" << i << ".png";
        Main_Log( INFO ) << "filename: " << ss.str();
        cv::Mat newImg = cv::imread( utils::findAbsoluteFilePath( ss.str() ), cv::IMREAD_GRAYSCALE );
        // auto t1        = std::chrono::high_resolution_clock::now();
        {
            TIMED_SCOPE( timernewImage, "New Image" );
            system.addImage( newImg, i );
        }
        // auto t2 = std::chrono::high_resolution_clock::now();
        // Main_Log( DEBUG ) << "Elapsed time for matching: " << std::chrono::duration_cast< std::chrono::milliseconds >( t2 - t1 ).count()
        //   << " milli sec";

        if ( config->m_enableVisualization == true )
        {
            if ( config->m_savingType == "LiveShow" )
            {
                cv::waitKey( 0 );
            }
        }
    }
    std::this_thread::sleep_for( std::chrono::seconds( 1 ) );
    // system.reportSummaryFrames();
    // system.reportSummaryFeatures();
    return 0;
}
