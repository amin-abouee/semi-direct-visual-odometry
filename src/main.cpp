#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <Eigen/Core>


// Efficient adaptive non-maximal suppression algorithms for homogeneous spatial keypoint distribution
Eigen::Matrix< double, 2, Eigen::Dynamic > Ssc(
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
    Eigen::Matrix< double, 2, Eigen::Dynamic > kp( 2, ResultVec.size() );
    for ( unsigned int i = 0; i < ResultVec.size(); i++ )
    {
        kp.col( i ) = Eigen::Vector2d( keyPoints[ ResultVec[ i ] ].pt.x, keyPoints[ ResultVec[ i ] ].pt.y );
    }
    return kp;
}

auto generateColor(const double min, const double max, const float value ) -> cv::Scalar
{
    int hue = (120 / (max-min)) * value;
    return cv::Scalar(hue, 80, 60);
}

int main(int argc, char* argv[])
{
    cv::Mat img = cv::imread("../input/0000000000.png", cv::IMREAD_GRAYSCALE);

    // https://answers.opencv.org/question/199237/most-accurate-visual-representation-of-gradient-magnitude/
    // https://answers.opencv.org/question/136622/how-to-calculate-gradient-in-c-using-opencv/
    // https://docs.opencv.org/master/d5/d0f/tutorial_py_gradients.html
    // https://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/sobel_derivatives/sobel_derivatives.html
    // http://ninghang.blogspot.com/2012/11/list-of-mat-type-in-opencv.html

    int ddepth = CV_32F;
    int ksize = 1;
    double scale = 1.0;
    double delta = 0.0;
    int borderType = cv::BORDER_DEFAULT;

    cv::Mat dx, absDx;
    cv::Sobel(img, dx, ddepth, 1, 0, ksize, scale, delta, borderType);
    cv::convertScaleAbs( dx, absDx );

    cv::Mat dy, absDy;
    cv::Sobel(img, dy, CV_32F, 0, 1, ksize, scale, delta, borderType);
    cv::convertScaleAbs( dy, absDy );


    // mag = cv2.magnitude(sobelx, sobely)  # so my Mat element values could be anything between 0 and 1???
    // ori = cv2.phase(sobelx, sobely, True) # so my Mat element values could be anything between 0 and 360 degrees???
    cv::Mat mag, angle;
    cv::cartToPolar(dx, dy, mag, angle, true);
    // std::cout << "mag type: " << mag.type() << std::endl;

    double min, max;
    cv::minMaxLoc(mag, &min, &max);
    // std::cout << "min: " << min << ", max: " << max << std::endl;

    // cv::Mat grad;
    // cv::addWeighted( absDx, 0.5, absDy, 0.5, 0, grad );
    // cv::imshow("grad_mag_weight", grad);

    // cv::Mat absoluteGrad = absDx + absDy;
    // cv::imshow("grad_mag_abs", absoluteGrad);

    // cv::Mat absMag;
    // cv::convertScaleAbs(mag, absMag);
    // cv::imshow("grad_mag_scale", absMag);

    cv::Mat normMag;
    cv::normalize(mag, normMag, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    cv::imshow("grad_mag_norm", normMag);

    cv::Mat histoMag;
    float range[] = { 0, 250 }; //the upper boundary is exclusive
    const float* histRange = { range };
    int histSize = 10;
    cv::calcHist(&mag, 1, 0, cv::Mat(), histoMag, 1, &histSize, &histRange);
    // cv::normalize(histoMag, histoMag, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    // cv::imshow("histo", histoMag);

    for(int i(0); i< histSize; i++)
        std::cout << "bin: " << i << ", val: " << histoMag.at<float>(i) << std::endl;

    std::vector< cv::KeyPoint > keyPoints;
    for(int i(0); i<img.rows; i++)
    {
        for(int j(0); j<img.cols; j++)
        {
            if (mag.at<float>(i, j) > 100.0)
            {
                keyPoints.emplace_back(cv::KeyPoint(cv::Point2f(j, i), 1.0, angle.at<float>(i, j), mag.at<float>(i, j)));
            }
        }
    }

    std::cout << "size keypoints: " << keyPoints.size() << std::endl;

    const int numRetPoints = 1000;
    const float tolerance = 0.1;
    const int cols = img.cols;
    const int rows = img.rows;

    const auto res = Ssc(keyPoints, numRetPoints, tolerance, cols, rows);
    std::cout << "select by ssc: " << res.cols() << std::endl;

    cv::Mat imgBGR;
    cv::cvtColor(normMag, imgBGR, cv::COLOR_GRAY2BGR);

    cv::Mat imgHSV;
    cv::cvtColor(imgBGR, imgHSV, cv::COLOR_BGR2HSV);


    for(int i(0); i< res.cols(); i++)
    {
        cv::circle(imgBGR, cv::Point2i(res.col(i)(0), res.col(i)(1)), 2.0, cv::Scalar(0, 255, 0));
    }
    cv::imshow("selected by ssc", imgBGR);


    // https://realpython.com/python-opencv-color-spaces/
    // https://stackoverflow.com/questions/23001512/c-and-opencv-get-and-set-pixel-color-to-mat
    // https://answers.opencv.org/question/178766/adjust-hue-and-saturation-like-photoshop/
    // http://colorizer.org/
    // https://toolstud.io/color/rgb.php?rgb_r=0&rgb_g=255&rgb_b=0&convert=rgbdec
    // https://answers.opencv.org/question/191488/create-a-hsv-range-palette/

    const float minMagnitude = 100.0;
    for(int i(0); i<imgHSV.rows; i++)
    {
        for(int j(0); j<imgHSV.cols; j++)
        {
            if (mag.at<float>(i, j) >= minMagnitude)
            {
                cv::Vec3b& px = imgHSV.at<cv::Vec3b>(cv::Point(j,i));
                cv::Scalar color = generateColor(minMagnitude, max, mag.at<float>(i, j) - minMagnitude );
                px[0] = color[0];
                px[1] = color[1];
                px[2] = color[2];
            }
        }
    }

    cv::Mat imgHSVNew;
    cv::cvtColor(imgHSV, imgHSVNew, cv::COLOR_HSV2BGR);
    cv::imshow("HSV", imgHSVNew);    
    // Vec3b color = image.at<Vec3b>(Point(x,y))

    // # create the mask and use it to change the colors
    // cv::inRange(imgHSV, lower, upper, imgHSV);

    cv::waitKey(0);
    return 0;
}
