#include "feature_selection.hpp"

#include <algorithm>
#include <vector>
#include <memory>
#include <cmath>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <Eigen/Core>

#include "easylogging++.h"
#define Feature_Log( LEVEL ) CLOG( LEVEL, "Feature" )

// #include "spdlog/sinks/stdout_color_sinks.h"

FeatureSelection::FeatureSelection(const cv::Mat& imgGray)
{
    // featureLogger = spdlog::stdout_color_mt( "FeatureSelection" );
    // featureLogger->set_level( spdlog::level::debug );
    // featureLogger->set_pattern( "[%Y-%m-%d %H:%M:%S] [%s:%#] [%n->%l] [thread:%t] %v" );

    // https://answers.opencv.org/question/199237/most-accurate-visual-representation-of-gradient-magnitude/
    // https://answers.opencv.org/question/136622/how-to-calculate-gradient-in-c-using-opencv/
    // https://docs.opencv.org/master/d5/d0f/tutorial_py_gradients.html
    // https://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/sobel_derivatives/sobel_derivatives.html
    // http://ninghang.blogspot.com/2012/11/list-of-mat-type-in-opencv.html

    int ddepth     = CV_32F;
    int ksize      = 1;
    double scale   = 1.0;
    double delta   = 0.0;
    int borderType = cv::BORDER_DEFAULT;

    // const cv::Mat imgGray = frame.m_imagePyramid.getBaseImage();
    auto t1 = std::chrono::high_resolution_clock::now();
    // cv::Mat dx, absDx;
    cv::Sobel( imgGray, m_dx, ddepth, 1, 0, ksize, scale, delta, borderType );
    // cv::convertScaleAbs( dx, absDx );

    // cv::Mat dy, absDy;
    cv::Sobel( imgGray, m_dy, CV_32F, 0, 1, ksize, scale, delta, borderType );


    // m_dx = cv::Mat ( imgGray.size(), CV_32F );
    // m_dy = cv::Mat ( imgGray.size(), CV_32F );
    // computeGradient(imgGray, m_dx, m_dy);

    // cv::Mat mag, angle;
    cv::cartToPolar( m_dx, m_dy, m_gradientMagnitude, m_gradientOrientation, true );
    auto t2 = std::chrono::high_resolution_clock::now();
    // std::cout << "Elapsed time for gradient magnitude: " << std::chrono::duration_cast< std::chrono::milliseconds >( t2 - t1 ).count()
            //   << std::endl;

    // featureLogger->info("Elapsed time for gradient magnitude: {}", std::chrono::duration_cast< std::chrono::milliseconds >( t2 - t1 ).count());

    Feature_Log(DEBUG) << "Init Feature Selection";
}

// FeatureSelection::FeatureSelection( const cv::Mat& imgGray )
// {
//     m_imgGray = std::make_shared< cv::Mat >( imgGray );
//     m_features.reserve(2000);
// }

// FeatureSelection::FeatureSelection( const cv::Mat& imgGray, const uint32_t numberFeatures ):
// m_numberFeatures(numberFeatures)
// {
//     m_imgGray = std::make_shared< cv::Mat >( imgGray );
//     m_features.reserve(numberFeatures * 2);
// }

void FeatureSelection::Ssc( std::shared_ptr<Frame>& frame,
                            const std::vector< cv::KeyPoint >& keyPoints,
                            const uint32_t numRetPoints,
                            const float tolerance,
                            const uint32_t cols,
                            const uint32_t rows )
{
    // several temp expression variables to simplify solution equation
    uint32_t exp1       = rows + cols + 2 * numRetPoints;
    long long exp2 = ( (long long)4 * cols + (long long)4 * numRetPoints + (long long)4 * rows * numRetPoints +
                       (long long)rows * rows + (long long)cols * cols - (long long)2 * rows * cols +
                       (long long)4 * rows * cols * numRetPoints );
    double exp3    = sqrt( static_cast<double>(exp2 ));
    double exp4    = ( 2 * ( numRetPoints - 1 ) );

    double sol1 = -round( ( exp1 + exp3 ) / exp4 );  // first solution
    double sol2 = -round( ( exp1 - exp3 ) / exp4 );  // second solution

    int high = ( sol1 > sol2 ) ? static_cast<int>(sol1) : static_cast<int>(sol2);  // binary search range initialization with positive solution
    int low  = static_cast<int>( sqrt( (double)keyPoints.size() / numRetPoints ) );

    int width;
    int prevWidth = -1;

    std::vector< uint32_t > ResultVec;
    bool complete     = false;
    float K    = static_cast<float>(numRetPoints);
    uint32_t Kmin = static_cast<uint32_t>(round( K - ( K * tolerance ) ));
    uint32_t Kmax = static_cast<uint32_t>(round( K + ( K * tolerance ) ));

    std::vector< uint32_t > result;
    result.reserve( keyPoints.size() );
    while ( !complete )
    {
        width = low + ( high - low ) / 2;
        if ( width == prevWidth || low > high )
        {                        // needed to reassure the same radius is not repeated again
            ResultVec = result;  // return the keypoints from the previous iteration
            break;
        }
        result.clear();
        double c        = width / 2.0;  // initializing Grid
        uint32_t numCellCols = static_cast<uint32_t>( cols / c );
        uint32_t numCellRows = static_cast<uint32_t>( rows / c );
        std::vector< std::vector< bool > > coveredVec( numCellRows + 1, std::vector< bool >( numCellCols + 1, false ) );

        for ( unsigned int i = 0; i < keyPoints.size(); ++i )
        {
            uint32_t row = static_cast<uint32_t>( keyPoints[ i ].pt.y / c );  // get position of the cell current point is located at
            uint32_t col = static_cast<uint32_t>( keyPoints[ i ].pt.x / c );
            if ( coveredVec[ row ][ col ] == false )
            {  // if the cell is not covered
                result.push_back( i );
                uint32_t rowMin = row >= static_cast<uint32_t>( width / c )  ? ( row - static_cast<uint32_t>( width / c ) )
                                                                   : 0;  // get range which current radius is covering
                uint32_t rowMax =
                  ( ( row + static_cast<uint32_t>( width / c ) ) <= numCellRows ) ? ( row + static_cast<uint32_t>( width / c ) ) : numCellRows;
                uint32_t colMin = col >= static_cast<uint32_t>( width / c )  ? ( col - static_cast<uint32_t>( width / c ) ) : 0;
                uint32_t colMax =
                  ( ( col + static_cast<uint32_t>( width / c ) ) <= numCellCols ) ? ( col + static_cast<uint32_t>( width / c ) ) : numCellCols;
                for ( uint32_t rowToCov = rowMin; rowToCov <= rowMax; ++rowToCov )
                {
                    for ( uint32_t colToCov = colMin; colToCov <= colMax; ++colToCov )
                    {
                        if ( !coveredVec[ rowToCov ][ colToCov ] )
                            coveredVec[ rowToCov ][ colToCov ] =
                              true;  // cover cells within the square bounding box with width w
                    }
                }
            }
        }

        if ( result.size() >= Kmin && result.size() <= Kmax )
        {  // solution found
            ResultVec = result;
            complete  = true;
        }
        else if ( result.size() < Kmin )
            high = width - 1;  // update binary search range
        else
            low = width + 1;
        prevWidth = width;
    }

    for ( unsigned int i = 0; i < ResultVec.size(); i++ )
    {
        const auto& kp = keyPoints[ ResultVec[ i ] ];
        std::unique_ptr< Feature > feature = std::make_unique< Feature >(
          frame, Eigen::Vector2d( kp.pt.x, kp.pt.y ), kp.response, kp.angle, 0 );
        frame->addFeature(feature);
    }
}

void FeatureSelection::detectFeaturesWithSSC( std::shared_ptr<Frame>& frame, const uint32_t numberCandidate )
{
    // const cv::Mat imgGray = frame.m_imagePyramid.getBaseImage();
    const int width  = frame->m_camera->width();
    const int height = frame->m_camera->height();

    std::vector< cv::KeyPoint > keyPoints;
    keyPoints.reserve( 10 * numberCandidate );
    for ( int i( 0 ); i < height; i++ )
    {
        for ( int j( 0 ); j < width; j++ )
        {
            if ( m_gradientMagnitude.at< float >( i, j ) > 75.0 )
            {
                keyPoints.emplace_back( cv::KeyPoint( cv::Point2i( j, i ), 1.0,
                                                      m_gradientOrientation.at< float >( i, j ),
                                                      m_gradientMagnitude.at< float >( i, j ) ) );
            }
        }
    }

    std::sort( keyPoints.begin(), keyPoints.end(),
               []( const cv::KeyPoint& lhs, const cv::KeyPoint& rhs ) { return lhs.response > rhs.response; } );

    // const int numRetPoints = 500;
    const float tolerance = 0.1f;
    Ssc( frame, keyPoints, numberCandidate, tolerance, static_cast<uint32_t>(width), static_cast<uint32_t>(height) );
}


void FeatureSelection::detectFeaturesInGrid( std::shared_ptr<Frame>& frame, const uint32_t gridSize )
{
    const uint32_t width  = frame->m_camera->width();
    const uint32_t height = frame->m_camera->height();

    const uint32_t cols = width / gridSize + 1;
    const uint32_t rows = height / gridSize + 1;

    // std::cout << "rows: " << rows << ", cols: " << cols << std::endl;

    // struct gridData
    // {
    //     int x;
    //     int y;
    //     float max;
    //     gridData()
    //     {
    //         x = -1;
    //         y = -1;
    //         max = 0.0;
    //     }
    // };

    // std::vector <gridData> stack(cols);
    // std::vector < std::vector <gridData> > table(rows, stack);

    // float* pixelPtr = m_gradientMagnitude.ptr<float>();
    // for(int i(0); i< m_gradientMagnitude.rows; i++)
    // {
    //     for(int j(0); j< m_gradientMagnitude.cols; j++, pixelPtr++)
    //     {
    //         const int indy = i / gridSize;
    //         const int indx = j / gridSize;
    //         if (*pixelPtr > table[indy][indx].max)
    //         {
    //             table[indy][indx].max = *pixelPtr;
    //             table[indy][indx].x = j;
    //             table[indy][indx].y = i;
    //         }
    //     }
    // }

    // for(int i(0); i<table.size(); i++)
    // {
    //     for (int j(0); j<table[i].size(); j++)
    //     {
    //         // std::cout << "row id: " << table[i][j].y << ", col id: " << table[i][j].x << ", max: " <<  table[i][j].max << std::endl; 
    //         const auto x = table[i][j].x;
    //         const auto y = table[i][j].y;
    //         const auto max = table[i][j].max;
    //         if (max > 0)
    //         {
    //             std::unique_ptr< Feature > feature = std::make_unique< Feature >(
    //                 frame, Eigen::Vector2d( x, y ), 
    //                 m_gradientMagnitude.at< float >( y, x ),  
    //                 m_gradientOrientation.at< float >( y, x ), 0 );
    //             frame.addFeature(feature);
    //         }
    //     }
    // }
    
    for (uint32_t r(0); r < rows; r++)
    {
        for (uint32_t c(0); c < cols; c++)
        {
            const uint32_t maxColIdx = (c+1) * gridSize < width ? gridSize : width - (c * gridSize);
            const uint32_t maxROwIdx = (r+1) * gridSize < height ? gridSize : height - (r * gridSize);
            const cv::Rect PatchROI( c * gridSize, r * gridSize, maxColIdx, maxROwIdx );
            // std::cout << "left corner: [" << r * gridSize << " , " << c * gridSize << "] -> ";
            const cv::Mat gradientPatch = m_gradientMagnitude( PatchROI );
            float max = 0.0;
            uint32_t rowIdx = 0;
            uint32_t colIdx = 0;
            for ( uint32_t i( 0 ); i < maxROwIdx; i++ )
            {
                for ( uint32_t j( 0 ); j < maxColIdx; j++ )
                {
                    if (gradientPatch.at< float >( i, j ) > max)
                    {
                        rowIdx = r * gridSize + i;
                        colIdx = c * gridSize + j;
                        max = gradientPatch.at< float >( i, j );
                    }
                }
            }

            // std::cout << "row id: " << rowIdx << ", col id: " << colIdx << ", max: " <<  max << std::endl;
            if (max > 0.0)
            {
                std::unique_ptr< Feature > feature = std::make_unique< Feature >(
                        frame, Eigen::Vector2d( colIdx, rowIdx ), 
                        m_gradientMagnitude.at< float >( rowIdx, colIdx ),  
                        m_gradientOrientation.at< float >( rowIdx, colIdx ), 0 );
                frame->addFeature(feature);
            }
        }
    }
    
}

void FeatureSelection::detectFeaturesByValue( std::shared_ptr<Frame>& frame, const uint32_t value )
{
    const uint32_t width  = frame->m_camera->width();
    const uint32_t height = frame->m_camera->height();

    for (uint32_t r(0); r < height; r++)
    {
        for (uint32_t c(0); c < width; c++)
        {
            if (m_gradientMagnitude.at< float >( r, c ) > value)
            {
                std::unique_ptr< Feature > feature = std::make_unique< Feature >(
                        frame, Eigen::Vector2d( c, r ), 
                        m_gradientMagnitude.at< float >( r, c ),  
                        m_gradientOrientation.at< float >( r, c ), 0 );
                frame->addFeature(feature);
            }
        }
    }
}


// void FeatureSelection::computeGradient( const cv::Mat& currentTemplateImage,
//                                             cv::Mat& templateGradientX,
//                                             cv::Mat& templateGradientY )
// {
//     int h = currentTemplateImage.rows;
//     int w = currentTemplateImage.cols;

//     // ALIGNMENT_LOG( DEBUG ) << "h: " << h << ", w: " << w;
//     // [1, 1; w-1, h-1]
//     for ( int y( 1 ); y < h - 1; y++ )
//     {
//         for ( int x( 1 ); x < w - 1; x++ )
//         {
//             templateGradientX.at< float >( y, x ) =
//               0.5 * ( currentTemplateImage.at< float >( y, x + 1 ) - currentTemplateImage.at< float >( y, x - 1 ) );
//             templateGradientY.at< float >( y, x ) =
//               0.5 * ( currentTemplateImage.at< float >( y + 1, x ) - currentTemplateImage.at< float >( y - 1, x ) );
//         }
//     }

//     // ALIGNMENT_LOG( DEBUG ) << "center computed";

//     // for first and last rows
//     for ( int x( 1 ); x < w - 1; x++ )
//     {
//         templateGradientX.at< float >( 0, x ) =
//           0.5 * ( currentTemplateImage.at< float >( 0, x + 1 ) - currentTemplateImage.at< float >( 0, x - 1 ) );
//         templateGradientY.at< float >( 0, x ) =
//           0.5 * ( currentTemplateImage.at< float >( 1, x ) - currentTemplateImage.at< float >( 0, x ) );

//         templateGradientX.at< float >( h - 1, x ) =
//           0.5 * ( currentTemplateImage.at< float >( h - 1, x + 1 ) - currentTemplateImage.at< float >( h - 1, x - 1 ) );
//         templateGradientY.at< float >( h - 1, x ) =
//           0.5 * ( currentTemplateImage.at< float >( h - 1, x ) - currentTemplateImage.at< float >( h - 2, x ) );
//     }

//     // ALIGNMENT_LOG( DEBUG ) << "first and last rows";

//     // for first and last cols
//     for ( int y( 1 ); y < h - 1; y++ )
//     {
//         templateGradientX.at< float >( y, 0 ) =
//           0.5 * ( currentTemplateImage.at< float >( y, 1 ) - currentTemplateImage.at< float >( y, 0 ) );
//         templateGradientY.at< float >( y, 0 ) =
//           0.5 * ( currentTemplateImage.at< float >( y + 1, 0 ) - currentTemplateImage.at< float >( y - 1, 0 ) );

//         templateGradientX.at< float >( y, w - 1 ) =
//           0.5 * ( currentTemplateImage.at< float >( y, w - 1 ) - currentTemplateImage.at< float >( y, w - 2 ) );
//         templateGradientY.at< float >( y, w - 1 ) =
//           0.5 * ( currentTemplateImage.at< float >( y + 1, w - 1 ) - currentTemplateImage.at< float >( y - 1, w - 1 ) );
//     }

//     // ALIGNMENT_LOG( DEBUG ) << "first and last cols";

//     // upper left
//     templateGradientX.at< float >( 0, 0 ) =
//       0.5 * ( currentTemplateImage.at< float >( 0, 1 ) - currentTemplateImage.at< float >( 0, 0 ) );
//     // upper right
//     templateGradientX.at< float >( 0, w - 1 ) =
//       0.5 * ( currentTemplateImage.at< float >( 0, w - 1 ) - currentTemplateImage.at< float >( 0, w - 2 ) );
//     // lower left
//     templateGradientX.at< float >( h - 1, 0 ) =
//       0.5 * ( currentTemplateImage.at< float >( h - 1, 1 ) - currentTemplateImage.at< float >( h - 1, 0 ) );
//     // lower right
//     templateGradientX.at< float >( h - 1, w - 1 ) =
//       0.5 * ( currentTemplateImage.at< float >( h - 1, w - 1 ) - currentTemplateImage.at< float >( h - 1, w - 2 ) );

//     // upper left
//     templateGradientY.at< float >( 0, 0 ) =
//       0.5 * ( currentTemplateImage.at< float >( 1, 0 ) - currentTemplateImage.at< float >( 0, 0 ) );
//     // upper right
//     templateGradientY.at< float >( 0, w - 1 ) =
//       0.5 * ( currentTemplateImage.at< float >( 1, w - 1 ) - currentTemplateImage.at< float >( 0, w - 1 ) );
//     // lower left
//     templateGradientY.at< float >( h - 1, 0 ) =
//       0.5 * ( currentTemplateImage.at< float >( h - 1, 0 ) - currentTemplateImage.at< float >( h - 2, 0 ) );
//     // lower right
//     templateGradientY.at< float >( h - 1, w - 1 ) =
//       0.5 * ( currentTemplateImage.at< float >( h - 1, w - 1 ) - currentTemplateImage.at< float >( h - 2, w - 1 ) );
// }