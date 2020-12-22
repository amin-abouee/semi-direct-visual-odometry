#include "feature_selection.hpp"

#include <Eigen/Core>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#define SIMD_OPENCV_ENABLE
#include <Simd/SimdLib.hpp>
#include <Simd/SimdView.hpp>

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

#include "easylogging++.h"
#define Feature_Log( LEVEL ) CLOG( LEVEL, "Feature" )

FeatureSelection::FeatureSelection( const int32_t width, const int32_t height, const int32_t cellSize )
    : m_cellSize( cellSize )
    , m_gridRows( height / cellSize + 1 )
    , m_gridCols( width / cellSize + 1 )
    , m_occupancyGrid( m_gridRows * m_gridCols, false )
{
}

void FeatureSelection::gradientMagnitudeWithSSC( std::shared_ptr< Frame >& frame,
                                                 const uint32_t detectionThreshold,
                                                 const uint32_t numberCandidate,
                                                 const bool useBucketing )
{
    const int32_t width  = frame->m_camera->width();
    const int32_t height = frame->m_camera->height();

    const cv::Mat& imgGray = frame->m_imagePyramid.getBaseImage();
    computeImageGradient( imgGray );

    // select all pixels with gradient magnitute > detectionThreshold
    std::vector< cv::KeyPoint > keyPoints;
    keyPoints.reserve( 10 * numberCandidate );
    for ( int32_t i( 0 ); i < height; i++ )
    {
        for ( int32_t j( 0 ); j < width; j++ )
        {
            if ( m_imgGradientMagnitude.at< uint8_t >( i, j ) > detectionThreshold )
            {
                keyPoints.emplace_back( cv::KeyPoint( cv::Point2i( j, i ), 1.0, m_imgGradientOrientation.at< uint8_t >( i, j ),
                                                      m_imgGradientMagnitude.at< uint8_t >( i, j ) ) );
            }
        }
    }

    // sort points based on their response (gradient magnitute)
    std::sort( keyPoints.begin(), keyPoints.end(),
               []( const cv::KeyPoint& lhs, const cv::KeyPoint& rhs ) { return lhs.response > rhs.response; } );

    const float tolerance = 0.1f;
    std::vector< int32_t > resultVec;
    resultVec.reserve( keyPoints.size() );
    SSC( keyPoints, numberCandidate, tolerance, width, height, resultVec );

    if ( useBucketing == true )
    {
        for ( unsigned int i = 0; i < resultVec.size(); i++ )
        {
            const auto& kp = keyPoints[ resultVec[ i ] ];
            int32_t idx    = static_cast< int32_t >( kp.pt.x ) / m_cellSize;
            int32_t idy    = static_cast< int32_t >( kp.pt.y ) / m_cellSize;
            if ( m_occupancyGrid[ idy * m_gridCols + idx ] == false )
            {
                m_occupancyGrid[ idy * m_gridCols + idx ] = true;
                std::shared_ptr< Feature > feature = std::make_shared< Feature >( frame, Eigen::Vector2d( kp.pt.x, kp.pt.y ), kp.response,
                                                                                  kp.angle, 0, Feature::FeatureType::EDGE );
                frame->addFeature( feature );
            }
        }

        resetGridOccupancy();
    }
    else
    {
        for ( unsigned int i = 0; i < resultVec.size(); i++ )
        {
            const auto& kp                     = keyPoints[ resultVec[ i ] ];
            std::shared_ptr< Feature > feature = std::make_shared< Feature >( frame, Eigen::Vector2d( kp.pt.x, kp.pt.y ), kp.response,
                                                                              kp.angle, 0, Feature::FeatureType::EDGE );
            frame->addFeature( feature );
        }
    }
}

void FeatureSelection::gradientMagnitudeByValue( std::shared_ptr< Frame >& frame,
                                                 const uint32_t detectionThreshold,
                                                 const bool useBucketing )
{
    const int32_t width  = frame->m_camera->width();
    const int32_t height = frame->m_camera->height();

    const cv::Mat& imgGray = frame->m_imagePyramid.getBaseImage();
    computeImageGradient( imgGray );

    if ( useBucketing == true )
    {
        // for over all cells
        for ( int32_t r( 0 ); r < m_gridRows; r++ )
        {
            for ( int32_t c( 0 ); c < m_gridCols; c++ )
            {
                if ( m_occupancyGrid[ r * m_gridCols + c ] == true )
                {
                    continue;
                }
                // get the border, image data, image gradient magnitute region of desire cell
                // compare the cell width and height border with image width and height
                const int32_t maxColIdx = ( c + 1 ) * m_cellSize < width ? m_cellSize : width - ( c * m_cellSize );
                const int32_t maxROwIdx = ( r + 1 ) * m_cellSize < height ? m_cellSize : height - ( r * m_cellSize );
                const cv::Rect PatchROI( c * m_cellSize, r * m_cellSize, maxColIdx, maxROwIdx );
                const cv::Mat gradientPatch = m_imgGradientMagnitude( PatchROI );
                uint32_t max                 = 0;
                int32_t rowIdx              = 0;
                int32_t colIdx              = 0;
                // for over all pixels of a cell
                for ( int32_t i( 0 ); i < maxROwIdx; i++ )
                {
                    for ( int32_t j( 0 ); j < maxColIdx; j++ )
                    {
                        if ( gradientPatch.at< uint8_t >( i, j ) > max )
                        {
                            rowIdx = r * m_cellSize + i;
                            colIdx = c * m_cellSize + j;
                            max    = gradientPatch.at< uint8_t >( i, j );
                        }
                    }
                }

                if ( max > detectionThreshold )
                {
                    std::shared_ptr< Feature > feature = std::make_shared< Feature >(
                      frame, Eigen::Vector2d( colIdx, rowIdx ), m_imgGradientMagnitude.at< uint8_t >( rowIdx, colIdx ),
                      m_imgGradientOrientation.at< uint8_t >( rowIdx, colIdx ), 0, Feature::FeatureType::EDGE );
                    frame->addFeature( feature );
                }
            }
        }

        resetGridOccupancy();
    }
    else
    {
        for ( int32_t r( 0 ); r < height; r++ )
        {
            for ( int32_t c( 0 ); c < width; c++ )
            {
                if ( m_imgGradientMagnitude.at< float >( r, c ) > detectionThreshold )
                {
                    std::shared_ptr< Feature > feature =
                      std::make_shared< Feature >( frame, Eigen::Vector2d( c, r ), m_imgGradientMagnitude.at< uint8_t >( r, c ),
                                                   m_imgGradientOrientation.at< uint8_t >( r, c ), 0, Feature::FeatureType::EDGE );
                    frame->addFeature( feature );
                }
            }
        }
    }
}

void FeatureSelection::SSC( const std::vector< cv::KeyPoint >& keyPoints,
                            const int32_t numRetPoints,
                            const float tolerance,
                            const int32_t cols,
                            const int32_t rows,
                            std::vector< int32_t >& resultVec )
{
    // several temp expression variables to simplify solution equation
    int32_t exp1   = rows + cols + 2 * numRetPoints;
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

    bool complete = false;
    float K       = static_cast< float >( numRetPoints );
    uint32_t Kmin = static_cast< uint32_t >( round( K - ( K * tolerance ) ) );
    uint32_t Kmax = static_cast< uint32_t >( round( K + ( K * tolerance ) ) );

    std::vector< int32_t > result;
    result.reserve( keyPoints.size() );
    while ( !complete )
    {
        width = low + ( high - low ) / 2;
        if ( width == prevWidth || low > high )
        {                        // needed to reassure the same radius is not repeated again
            resultVec = result;  // return the keypoints from the previous iteration
            break;
        }
        result.clear();
        double c            = width / 2.0;  // initializing Grid
        int32_t numCellCols = static_cast< int32_t >( cols / c );
        int32_t numCellRows = static_cast< int32_t >( rows / c );
        std::vector< std::vector< bool > > coveredVec( numCellRows + 1, std::vector< bool >( numCellCols + 1, false ) );

        for ( unsigned int i = 0; i < keyPoints.size(); ++i )
        {
            int32_t row = static_cast< int32_t >( keyPoints[ i ].pt.y / c );  // get position of the cell current point is located at
            int32_t col = static_cast< int32_t >( keyPoints[ i ].pt.x / c );
            if ( coveredVec[ row ][ col ] == false )
            {  // if the cell is not covered
                result.push_back( i );
                int32_t rowMin = row >= static_cast< int32_t >( width / c ) ? ( row - static_cast< int32_t >( width / c ) )
                                                                            : 0;  // get range which current radius is covering
                int32_t rowMax = ( ( row + static_cast< int32_t >( width / c ) ) <= numCellRows )
                                   ? ( row + static_cast< int32_t >( width / c ) )
                                   : numCellRows;
                int32_t colMin = col >= static_cast< int32_t >( width / c ) ? ( col - static_cast< int32_t >( width / c ) ) : 0;
                int32_t colMax = ( ( col + static_cast< int32_t >( width / c ) ) <= numCellCols )
                                   ? ( col + static_cast< int32_t >( width / c ) )
                                   : numCellCols;
                for ( int32_t rowToCov = rowMin; rowToCov <= rowMax; ++rowToCov )
                {
                    for ( int32_t colToCov = colMin; colToCov <= colMax; ++colToCov )
                    {
                        if ( !coveredVec[ rowToCov ][ colToCov ] )
                            coveredVec[ rowToCov ][ colToCov ] = true;  // cover cells within the square bounding box with width w
                    }
                }
            }
        }

        if ( result.size() >= Kmin && result.size() <= Kmax )
        {  // solution found
            resultVec = result;
            complete  = true;
        }
        else if ( result.size() < Kmin )
            high = width - 1;  // update binary search range
        else
            low = width + 1;
        prevWidth = width;
    }
}

void FeatureSelection::computeImageGradient( const cv::Mat& imgGray )
{
    // https://answers.opencv.org/question/199237/most-accurate-visual-representation-of-gradient-magnitude/
    // https://answers.opencv.org/question/136622/how-to-calculate-gradient-in-c-using-opencv/
    // https://docs.opencv.org/master/d5/d0f/tutorial_py_gradients.html
    // https://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/sobel_derivatives/sobel_derivatives.html
    // http://ninghang.blogspot.com/2012/11/list-of-mat-type-in-opencv.html

    TIMED_FUNC( timerObj );

    m_imgGradientMagnitude   = cv::Mat( imgGray.size(), CV_8U );
    m_imgGradientOrientation = cv::Mat( imgGray.size(), CV_8U, cv::Scalar( 0 ) );

    Simd::View< Simd::Allocator > src = imgGray;
    Simd::View< Simd::Allocator > dst = m_imgGradientMagnitude;
    Simd::AbsGradientSaturatedSum( src, dst );
    // Simd::PrintInfo(std::cout);

}

void FeatureSelection::setExistingFeatures( const std::vector< std::shared_ptr< Feature > >& features )
{
    for ( const auto& feature : features )
    {
        setCellInGridOccupancy( feature->m_pixelPosition );
    }
}

void FeatureSelection::setCellInGridOccupancy( const Eigen::Vector2d& location )
{
    uint32_t idx                              = location.x() / m_cellSize;
    uint32_t idy                              = location.y() / m_cellSize;
    m_occupancyGrid[ idy * m_gridCols + idx ] = true;
}

void FeatureSelection::resetGridOccupancy()
{
    std::fill( m_occupancyGrid.begin(), m_occupancyGrid.end(), false );
}
