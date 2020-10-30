#include "feature_selection.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <Eigen/Core>

#include "easylogging++.h"
#define Feature_Log( LEVEL ) CLOG( LEVEL, "Feature" )

FeatureSelection::FeatureSelection( const int32_t width, const int32_t height, const uint32_t cellSize )
    : m_cellSize( cellSize )
    , m_gridRows( height / cellSize + 1 )
    , m_gridCols( width / cellSize + 1 )
    , m_occupancyGrid( m_gridRows * m_gridCols, false )
{
}

void FeatureSelection::detectFeaturesWithSSC( std::shared_ptr< Frame >& frame, const uint32_t numberCandidate )
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
                keyPoints.emplace_back( cv::KeyPoint( cv::Point2i( j, i ), 1.0, m_gradientOrientation.at< float >( i, j ),
                                                      m_gradientMagnitude.at< float >( i, j ) ) );
            }
        }
    }

    std::sort( keyPoints.begin(), keyPoints.end(),
               []( const cv::KeyPoint& lhs, const cv::KeyPoint& rhs ) { return lhs.response > rhs.response; } );

    // const int numRetPoints = 500;
    const float tolerance = 0.1f;
    Ssc( frame, keyPoints, numberCandidate, tolerance, static_cast< uint32_t >( width ), static_cast< uint32_t >( height ) );
}

void FeatureSelection::detectFeaturesInGrid( std::shared_ptr< Frame >& frame, const float detectionThreshold )
{
    const int width  = frame->m_camera->width();
    const int height = frame->m_camera->height();

    cv::Mat imgGradientMagnitude;
    cv::Mat imgGradientOrientation;
    const cv::Mat& imgGray = frame->m_imagePyramid.getBaseImage();
    comouteImageGradient (imgGray, imgGradientMagnitude, imgGradientOrientation);

    // std::cout << "rows: " << rows << ", cols: " << cols << std::endl;

    for ( uint32_t r( 0 ); r < m_gridRows; r++ )
    {
        for ( uint32_t c( 0 ); c < m_gridCols; c++ )
        {
            if ( m_occupancyGrid[r * m_gridCols + c] == true)
            {
                continue;
            }

            const uint32_t maxColIdx = ( c + 1 ) * m_cellSize < width ? m_cellSize : width - ( c * m_cellSize );
            const uint32_t maxROwIdx = ( r + 1 ) * m_cellSize < height ? m_cellSize : height - ( r * m_cellSize );
            const cv::Rect PatchROI( c * m_cellSize, r * m_cellSize, maxColIdx, maxROwIdx );
            // std::cout << "left corner: [" << r * m_cellSize << " , " << c * m_cellSize << "] -> ";
            const cv::Mat gradientPatch = imgGradientMagnitude( PatchROI );
            float max                   = 0.0;
            uint32_t rowIdx             = 0;
            uint32_t colIdx             = 0;
            for ( uint32_t i( 0 ); i < maxROwIdx; i++ )
            {
                for ( uint32_t j( 0 ); j < maxColIdx; j++ )
                {
                    if ( gradientPatch.at< float >( i, j ) > max )
                    {
                        rowIdx = r * m_cellSize + i;
                        colIdx = c * m_cellSize + j;
                        max    = gradientPatch.at< float >( i, j );
                    }
                }
            }

            // std::cout << "row id: " << rowIdx << ", col id: " << colIdx << ", max: " <<  max << std::endl;
            if ( max > detectionThreshold )
            {
                std::shared_ptr< Feature > feature =
                  std::make_shared< Feature >( frame, Eigen::Vector2d( colIdx, rowIdx ), imgGradientMagnitude.at< float >( rowIdx, colIdx ),
                                               imgGradientOrientation.at< float >( rowIdx, colIdx ), 0 );
                frame->addFeature( feature );
            }
        }
    }

    resetGridOccupancy();
}

void FeatureSelection::detectFeaturesByValue( std::shared_ptr< Frame >& frame, const float detectionThreshold )
{
    const uint32_t width  = frame->m_camera->width();
    const uint32_t height = frame->m_camera->height();

    for ( uint32_t r( 0 ); r < height; r++ )
    {
        for ( uint32_t c( 0 ); c < width; c++ )
        {
            if ( m_gradientMagnitude.at< float >( r, c ) > detectionThreshold )
            {
                std::shared_ptr< Feature > feature = std::make_shared< Feature >(
                  frame, Eigen::Vector2d( c, r ), m_gradientMagnitude.at< float >( r, c ), m_gradientOrientation.at< float >( r, c ), 0 );
                frame->addFeature( feature );
            }
        }
    }
}

void FeatureSelection::Ssc( std::shared_ptr< Frame >& frame,
                            const std::vector< cv::KeyPoint >& keyPoints,
                            const uint32_t numRetPoints,
                            const float tolerance,
                            const uint32_t cols,
                            const uint32_t rows )
{
    // several temp expression variables to simplify solution equation
    uint32_t exp1  = rows + cols + 2 * numRetPoints;
    long long exp2 = ( (long long)4 * cols + (long long)4 * numRetPoints + (long long)4 * rows * numRetPoints + (long long)rows * rows +
                       (long long)cols * cols - (long long)2 * rows * cols + (long long)4 * rows * cols * numRetPoints );
    double exp3    = sqrt( static_cast< double >( exp2 ) );
    double exp4    = ( 2 * ( numRetPoints - 1 ) );

    double sol1 = -round( ( exp1 + exp3 ) / exp4 );  // first solution
    double sol2 = -round( ( exp1 - exp3 ) / exp4 );  // second solution

    int high = ( sol1 > sol2 ) ? static_cast< int >( sol1 )
                               : static_cast< int >( sol2 );  // binary search range initialization with positive solution
    int low = static_cast< int >( sqrt( (double)keyPoints.size() / numRetPoints ) );

    int width;
    int prevWidth = -1;

    std::vector< uint32_t > ResultVec;
    bool complete = false;
    float K       = static_cast< float >( numRetPoints );
    uint32_t Kmin = static_cast< uint32_t >( round( K - ( K * tolerance ) ) );
    uint32_t Kmax = static_cast< uint32_t >( round( K + ( K * tolerance ) ) );

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
        double c             = width / 2.0;  // initializing Grid
        uint32_t numCellCols = static_cast< uint32_t >( cols / c );
        uint32_t numCellRows = static_cast< uint32_t >( rows / c );
        std::vector< std::vector< bool > > coveredVec( numCellRows + 1, std::vector< bool >( numCellCols + 1, false ) );

        for ( unsigned int i = 0; i < keyPoints.size(); ++i )
        {
            uint32_t row = static_cast< uint32_t >( keyPoints[ i ].pt.y / c );  // get position of the cell current point is located at
            uint32_t col = static_cast< uint32_t >( keyPoints[ i ].pt.x / c );
            if ( coveredVec[ row ][ col ] == false )
            {  // if the cell is not covered
                result.push_back( i );
                uint32_t rowMin = row >= static_cast< uint32_t >( width / c ) ? ( row - static_cast< uint32_t >( width / c ) )
                                                                              : 0;  // get range which current radius is covering
                uint32_t rowMax = ( ( row + static_cast< uint32_t >( width / c ) ) <= numCellRows )
                                    ? ( row + static_cast< uint32_t >( width / c ) )
                                    : numCellRows;
                uint32_t colMin = col >= static_cast< uint32_t >( width / c ) ? ( col - static_cast< uint32_t >( width / c ) ) : 0;
                uint32_t colMax = ( ( col + static_cast< uint32_t >( width / c ) ) <= numCellCols )
                                    ? ( col + static_cast< uint32_t >( width / c ) )
                                    : numCellCols;
                for ( uint32_t rowToCov = rowMin; rowToCov <= rowMax; ++rowToCov )
                {
                    for ( uint32_t colToCov = colMin; colToCov <= colMax; ++colToCov )
                    {
                        if ( !coveredVec[ rowToCov ][ colToCov ] )
                            coveredVec[ rowToCov ][ colToCov ] = true;  // cover cells within the square bounding box with width w
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
        std::shared_ptr< Feature > feature =
          std::make_shared< Feature >( frame, Eigen::Vector2d( kp.pt.x, kp.pt.y ), kp.response, kp.angle, 0 );
        frame->addFeature( feature );
    }
}

void FeatureSelection::comouteImageGradient(const cv::Mat& imgGray, cv::Mat& imgGradientMagnitude, cv::Mat& imgGradientOrientation)
{
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

    cv::Mat dx;
    cv::Mat dy;

    // const cv::Mat imgGray = frame.m_imagePyramid.getBaseImage();
    // auto t1 = std::chrono::high_resolution_clock::now();
    // cv::Mat dx, absDx;
    cv::Sobel( imgGray, dx, ddepth, 1, 0, ksize, scale, delta, borderType );
    // cv::convertScaleAbs( dx, absDx );

    // cv::Mat dy, absDy;
    cv::Sobel( imgGray, dy, ddepth, 0, 1, ksize, scale, delta, borderType );

    // m_dx = cv::Mat ( imgGray.size(), CV_32F );
    // m_dy = cv::Mat ( imgGray.size(), CV_32F );
    // computeGradient(imgGray, m_dx, m_dy);

    // cv::Mat mag, angle;
    cv::cartToPolar( dx, dy, imgGradientMagnitude, imgGradientOrientation, true );
}

void FeatureSelection::setExistingFeatures (const std::vector<std::shared_ptr<Feature>>& features)
{
    for(const auto& feature : features)
    {
        setCellInGridOccupancy(feature->m_feature);
    }
}


void FeatureSelection::setCellInGridOccupancy(const Eigen::Vector2d& location)
{
    uint32_t idx = location.x() / m_cellSize;
    uint32_t idy = location.y() / m_cellSize;
    m_occupancyGrid[idy * m_gridCols + idx] = true;
}

void FeatureSelection::resetGridOccupancy ()
{
    std::fill(m_occupancyGrid.begin(), m_occupancyGrid.end(), false);
}
