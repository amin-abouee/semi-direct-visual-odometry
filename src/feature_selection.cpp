#include "feature_selection.hpp"

#include <algorithm>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <Eigen/Core>

FeatureSelection::FeatureSelection( const cv::Mat& imgGray )
{
    m_imgGray = std::make_shared< cv::Mat >( imgGray );

    // cv::convertScaleAbs( dy, absDy );
}

Eigen::Matrix< double, 3, Eigen::Dynamic > FeatureSelection::Ssc(
  std::vector< cv::KeyPoint > keyPoints, int numRetPoints, float tolerance, int cols, int rows )
{
    // several temp expression variables to simplify solution equation
    int exp1       = rows + cols + 2 * numRetPoints;
    long long exp2 = ( (long long)4 * cols + (long long)4 * numRetPoints + (long long)4 * rows * numRetPoints +
                       (long long)rows * rows + (long long)cols * cols - (long long)2 * rows * cols +
                       (long long)4 * rows * cols * numRetPoints );
    double exp3    = sqrt( exp2 );
    double exp4    = ( 2 * ( numRetPoints - 1 ) );

    double sol1 = -round( ( exp1 + exp3 ) / exp4 );  // first solution
    double sol2 = -round( ( exp1 - exp3 ) / exp4 );  // second solution

    int high = ( sol1 > sol2 ) ? sol1 : sol2;  // binary search range initialization with positive solution
    int low  = floor( sqrt( (double)keyPoints.size() / numRetPoints ) );

    int width;
    int prevWidth = -1;

    std::vector< int > ResultVec;
    bool complete     = false;
    unsigned int K    = numRetPoints;
    unsigned int Kmin = round( K - ( K * tolerance ) );
    unsigned int Kmax = round( K + ( K * tolerance ) );

    std::vector< int > result;
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
        double c        = width / 2;  // initializing Grid
        int numCellCols = floor( cols / c );
        int numCellRows = floor( rows / c );
        std::vector< std::vector< bool > > coveredVec( numCellRows + 1, std::vector< bool >( numCellCols + 1, false ) );

        for ( unsigned int i = 0; i < keyPoints.size(); ++i )
        {
            int row = floor( keyPoints[ i ].pt.y / c );  // get position of the cell current point is located at
            int col = floor( keyPoints[ i ].pt.x / c );
            if ( coveredVec[ row ][ col ] == false )
            {  // if the cell is not covered
                result.push_back( i );
                int rowMin = ( ( row - floor( width / c ) ) >= 0 ) ? ( row - floor( width / c ) )
                                                                   : 0;  // get range which current radius is covering
                int rowMax =
                  ( ( row + floor( width / c ) ) <= numCellRows ) ? ( row + floor( width / c ) ) : numCellRows;
                int colMin = ( ( col - floor( width / c ) ) >= 0 ) ? ( col - floor( width / c ) ) : 0;
                int colMax =
                  ( ( col + floor( width / c ) ) <= numCellCols ) ? ( col + floor( width / c ) ) : numCellCols;
                for ( int rowToCov = rowMin; rowToCov <= rowMax; ++rowToCov )
                {
                    for ( int colToCov = colMin; colToCov <= colMax; ++colToCov )
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
    // retrieve final keypoints
    // std::vector< cv::KeyPoint > kp;
    Eigen::Matrix< double, 3, Eigen::Dynamic > kp( 3, ResultVec.size() );
    for ( unsigned int i = 0; i < ResultVec.size(); i++ )
    {
        kp.col( i ) = Eigen::Vector3d( keyPoints[ ResultVec[ i ] ].pt.x, keyPoints[ ResultVec[ i ] ].pt.y, 1.0 );
    }
    return kp;
}

void FeatureSelection::detectFeatures( const uint32_t numberCandidate )
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

    // cv::Mat dx, absDx;
    cv::Sobel( *m_imgGray, m_dx, ddepth, 1, 0, ksize, scale, delta, borderType );
    // cv::convertScaleAbs( dx, absDx );

    // cv::Mat dy, absDy;
    cv::Sobel( *m_imgGray, m_dy, CV_32F, 0, 1, ksize, scale, delta, borderType );

    // cv::Mat mag, angle;
    cv::cartToPolar( m_dx, m_dy, m_gradientMagnitude, m_gradientOrientation, true );

    std::vector< cv::KeyPoint > keyPoints;
    keyPoints.reserve( 10 * numberCandidate );
    for ( int i( 0 ); i < m_imgGray->rows; i++ )
    {
        for ( int j( 0 ); j < m_imgGray->cols; j++ )
        {
            if ( m_gradientMagnitude.at< float >( i, j ) > 75.0 )
            {
                keyPoints.emplace_back( cv::KeyPoint( cv::Point2f( j, i ), 1.0,
                                                      m_gradientOrientation.at< float >( i, j ),
                                                      m_gradientMagnitude.at< float >( i, j ) ) );
            }
        }
    }

    std::sort( keyPoints.begin(), keyPoints.end(),
               []( const cv::KeyPoint& lhs, const cv::KeyPoint& rhs ) { return lhs.response > rhs.response; } );

    // const int numRetPoints = 500;
    const float tolerance = 0.1;
    const int cols        = m_imgGray->cols;
    const int rows        = m_imgGray->rows;

    m_kp = Ssc( keyPoints, numberCandidate, tolerance, cols, rows );
}

// void FeatureSelection::visualizeFeaturePoints()
// {
//     cv::Mat normMag, imgBGR;
//     cv::normalize( m_gradientMagnitude, normMag, 0, 255, cv::NORM_MINMAX, CV_8UC1 );
//     cv::cvtColor( normMag, imgBGR, cv::COLOR_GRAY2BGR );

//     for ( int i( 0 ); i < m_kp.cols(); i++ )
//     {
//         cv::circle( imgBGR, cv::Point2i( m_kp.col( i )( 0 ), m_kp.col( i )( 1 ) ), 2.0, cv::Scalar( 0, 255, 0 ) );
//     }
//     cv::imshow( "selected by ssc", imgBGR );
// }

// void FeatureSelection::visualizeGrayGradientMagnitude()
// {
//     double min, max;
//     cv::minMaxLoc( m_gradientMagnitude, &min, &max );
//     // std::cout << "min: " << min << ", max: " << max << std::endl;

//     // cv::Mat grad;
//     // cv::addWeighted( absDx, 0.5, absDy, 0.5, 0, grad );
//     // cv::imshow("grad_mag_weight", grad);

//     // cv::Mat absoluteGrad = absDx + absDy;
//     // cv::imshow("grad_mag_abs", absoluteGrad);

//     // cv::Mat absMag;
//     // cv::convertScaleAbs(mag, absMag);
//     // cv::imshow("grad_mag_scale", absMag);

//     cv::Mat normMag;
//     cv::normalize( m_gradientMagnitude, normMag, 0, 255, cv::NORM_MINMAX, CV_8UC1 );
//     cv::imshow( "grad_mag_norm", normMag );
// }

// void FeatureSelection::visualizeColoredGradientMagnitude()
// {
//     // https://realpython.com/python-opencv-color-spaces/
//     // https://stackoverflow.com/questions/23001512/c-and-opencv-get-and-set-pixel-color-to-mat
//     // https://answers.opencv.org/question/178766/adjust-hue-and-saturation-like-photoshop/
//     // http://colorizer.org/
//     // https://toolstud.io/color/rgb.php?rgb_r=0&rgb_g=255&rgb_b=0&convert=rgbdec
//     // https://answers.opencv.org/question/191488/create-a-hsv-range-palette/

//     cv::Mat normMag, imgBGR, imgHSV;
//     cv::normalize( m_gradientMagnitude, normMag, 0, 255, cv::NORM_MINMAX, CV_8UC1 );
//     cv::cvtColor( normMag, imgBGR, cv::COLOR_GRAY2BGR );
//     cv::cvtColor( imgBGR, imgHSV, cv::COLOR_BGR2HSV );

//     const float minMagnitude = 75.0;
//     for ( int i( 0 ); i < imgHSV.rows; i++ )
//     {
//         for ( int j( 0 ); j < imgHSV.cols; j++ )
//         {
//             if ( m_gradientMagnitude.at< float >( i, j ) >= minMagnitude )
//             {
//                 cv::Vec3b& px = imgHSV.at< cv::Vec3b >( cv::Point( j, i ) );
//                 cv::Scalar color =
//                   generateColor( minMagnitude, 255.0, m_gradientMagnitude.at< float >( i, j ) - minMagnitude );
//                 px[ 0 ] = color[ 0 ];
//                 px[ 1 ] = color[ 1 ];
//                 px[ 2 ] = color[ 2 ];
//             }
//         }
//     }
//     cv::Mat imgHSVNew;
//     cv::cvtColor( imgHSV, imgHSVNew, cv::COLOR_HSV2BGR );
//     cv::imshow( "HSV", imgHSVNew );
// }

// cv::Scalar FeatureSelection::generateColor( const double min, const double max, const float value )
// {
//     int hue = ( 120 / ( max - min ) ) * value;
//     return cv::Scalar( hue, 100, 100 );
// }

// void FeatureSelection::visualizeEpipolar( const Eigen::Vector3d& point, const Eigen::Matrix3d& K )
// {
//     const Eigen::Vector3d projected = K * point;
//     cv::Mat normMag, imgBGR;
//     cv::normalize( m_gradientMagnitude, normMag, 0, 255, cv::NORM_MINMAX, CV_8UC1 );
//     cv::cvtColor( normMag, imgBGR, cv::COLOR_GRAY2BGR );
//     cv::circle( imgBGR, cv::Point2i( projected( 0 ), projected( 1 ) ), 5.0, cv::Scalar( 0, 255, 165 ) );
//     cv::imshow( "Epipolar", imgBGR );
// }