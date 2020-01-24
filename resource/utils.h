#ifndef ARTSYSTEM_SRC_TOOLS_POSECNN_UTILS_H_
#define ARTSYSTEM_SRC_TOOLS_POSECNN_UTILS_H_

#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>
#include <chrono>//#include <boost/chrono/thread_clock.hpp>
#include <memory>
#include <cmath>
#include <limits>
#include <type_traits>
#include <algorithm>
#include <numeric> // std::iota

#include <iomanip>

#include <opencv2/opencv.hpp>
/*
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
*/

// TODO: Split this file into OpenCV and mist utils

#ifndef M_PI
#define M_PI (4.0 * std::atan2(1.0, 1.0)) // M_PI is a non-standard macro
#endif


/**
 * @file
 * Just a few convenient utilities functions
 * All functions ought to be inlined or templated to avoid "multiple definitions" errors
 */

namespace PKutils
{

class Timer {

public:
    typedef std::chrono::high_resolution_clock Clock;
    //typedef std::chrono::steady_clock Clock;
    //typedef boost::chrono::thread_clock Clock;;
    typedef std::chrono::high_resolution_clock::duration duration;

    Timer()
    {
        m_paused = false;
    }

    void start() {
        if ( m_paused )
        {
            resume();
        } 
        else
        {
            t1 = Clock::now();
        }
    }

    void stop() {
        t2 = Clock::now();
        reset();
    }

    Clock::time_point pause() {
        assert(!m_paused);
        t2 = Clock::now();
        m_paused = true;
        return t2;
    }

    Clock::time_point resume() {
        assert(m_paused);
        m_paused = false;
        Clock::time_point now = Clock::now();
        t1 += now - t2;
        return now;
    }

    void reset() {
        m_paused = false;
    }

    int getElapsedMicroseconds() {
        return std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();
        //return boost::chrono::duration_cast<boost::chrono::microseconds>(t2-t1).count();
    }

    int getElapsedMilliseconds() {
        return std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count();
        //return boost::chrono::duration_cast<boost::chrono::microseconds>(t2-t1).count();
    }

    double getElapsedSeconds() {
        return std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1> > > (t2-t1).count();
        //return boost::chrono::duration_cast<boost::chrono::duration<double>>(t2-t1).count();
    }

    void report(const std::string &description) {
        printf( "%s: %d \u00B5s, %8.3f ms: %5.3f s\n", description.c_str(), getElapsedMicroseconds(),  getElapsedMicroseconds()/1000.0, getElapsedSeconds() );
    }

private:
    Clock::time_point t1;
    Clock::time_point t2;
    bool m_paused;

};


// printf-style formatting returning std::string
// Convenient, but neither particulary efficient nor typesafe
template<typename ... Args>
inline std::string string_printf( const std::string& format, Args ... args )
{
    std::size_t size = std::snprintf( nullptr, 0, format.c_str(), args ... ) + 1; // Don't forget '\0'
    std::unique_ptr<char[]> buf( new char[ size ] ); // Allocate on the heap
    std::snprintf( buf.get(), size, format.c_str(), args ... );
    return std::string( buf.get(), buf.get() + size - 1 ); // Remove trailing '\0'
}

template <>
inline std::string string_printf( const std::string& format ) { // avoids "not a string literal" error in case of sizeof...(args) == 0
    return format;
}



inline void printMatrix(cv::Mat &mat, std::string description) {
    std::string datatyp = "dont' know";
    if(mat.depth() == 5) {
        datatyp = "float";
    } else if(mat.depth() == 6) {
        datatyp = "double";
    }
    std::cout << std::endl << description << " (" << mat.rows <<"x" << mat.cols << " of type " << datatyp << ")" << std::endl;
    std::cout << std::setprecision(15) << std::right << std::fixed;

    if(mat.depth() == 5) { //data type = CV_32FC1
        for(int row = 0; row < mat.rows; row++) {
            for(int col = 0; col < mat.cols; col++) {
                std::cout << std::setw(20) << mat.at<float>(row, col);
            }
            std::cout << std::endl;
        }
    } else if (mat.depth() == 6) { //data type = CV_64FC1
        for(int row = 0; row < mat.rows; row++) {
            for(int col = 0; col < mat.cols; col++) {
                std::cout << std::setw(20) << mat.at<double>(row, col);
            }
            std::cout << std::endl;
        }
    } else {
        std::cout << "Matrix type not supported yet. Feel free to expand printMatrix()" <<std:: endl;
    }
    std::cout <<std:: endl;
}


// http://stackoverflow.com/questions/10167534/how-to-find-out-what-type-of-a-mat-object-is-with-mattype-in-opencv
// +--------+----+----+----+----+------+------+------+------+
// |        | C1 | C2 | C3 | C4 | C(5) | C(6) | C(7) | C(8) |
// +--------+----+----+----+----+------+------+------+------+
// | CV_8U  |  0 |  8 | 16 | 24 |   32 |   40 |   48 |   56 |
// | CV_8S  |  1 |  9 | 17 | 25 |   33 |   41 |   49 |   57 |
// | CV_16U |  2 | 10 | 18 | 26 |   34 |   42 |   50 |   58 |
// | CV_16S |  3 | 11 | 19 | 27 |   35 |   43 |   51 |   59 |
// | CV_32S |  4 | 12 | 20 | 28 |   36 |   44 |   52 |   60 |
// | CV_32F |  5 | 13 | 21 | 29 |   37 |   45 |   53 |   61 |
// | CV_64F |  6 | 14 | 22 | 30 |   38 |   46 |   54 |   62 |
// +--------+----+----+----+----+------+------+------+------+
// 
// Reference:
// New location: https://github.com/opencv/opencv/blob/master/modules/core/include/opencv2/core/hal/interface.h
// Old location: https://github.com/opencv/opencv/blob/master/modules/core/include/opencv2/core/types_c.h
inline std::string matrixInfoStr( const cv::Mat& M )
{
    
    // 1) The human-readable way
    //int type = M.type();
    int depth = M.depth();
    int channels = M.channels();
           
    // 2) The macro way
    //int type = M.type();
    //int depth = CV_MAT_DEPTH(type);
    //int channels = CV_MAT_CN(type);
    
    // 3) The bit twiddling way
    //int type = M.type();
    //unsigned char depth =  type & CV_MAT_DEPTH_MASK;
    //unsigned char C = 1 + (type >> CV_CN_SHIFT);
    ////C += '0'; // + 48 for ASCII printable characters. Only works for '1, 2, ... 9' channels
    //int channels = static_cast<int>(C);
    
    std::stringstream ss;
    switch ( depth ) {
        case CV_8U:  ss << "8U"; break;
        case CV_8S:  ss << "8S"; break;
        case CV_16U: ss << "16U"; break;
        case CV_16S: ss << "16S"; break;
        case CV_32S: ss << "32S"; break;
        case CV_32F: ss << "32F"; break;
        case CV_64F: ss << "64F"; break;
        default:     ss << "User"; break;
    }
    ss << "C" << channels << " " << M.rows << "x" << M.cols << "x" << channels << " (rows x cols x channels)"; 
    return ss.str();
}

template <typename T>
void convertKeypoints(const std::vector<cv::KeyPoint>& keypoints, std::vector<T>& points, const std::vector<int>& keypointIndexes=std::vector<int>())
{
    if( keypointIndexes.empty() )
    {
        points.resize( keypoints.size() );
        for( std::size_t i = 0; i < keypoints.size(); i++ )
            points[i] = keypoints[i].pt;
    }
    else
    {
        points.resize( keypointIndexes.size() );
        for( std::size_t i = 0; i < keypointIndexes.size(); i++ )
        {
            int idx = keypointIndexes[i];
            if( idx >= 0 )
                points[i] = keypoints[idx].pt;
        }
    }
}



/**
 * Floating point comparison:
 * "What Every Computer Scientist Should Know About Floating Point Arithmetic", David Goldberg (1991)
 * @details Units in the Last Place: The larger the value, the larger the tolerance.
 *          ULP 0: x == y
 *          ULP 4: Suitable for 80 bits precision
 *          Attention: this method is still not entirely fool proof but at least combines absolute and relative tolerances
 * @param x The first float.
 * @param y The second float.
 * @param ulp Units in the Last Place.
 * @return True if floats are almost equal.
 */
template<class T>
typename std::enable_if<!std::numeric_limits<T>::is_integer, bool>::type
almostEqual(T x, T y, int ulp = 4)
{
    return std::abs(x-y) < std::numeric_limits<T>::epsilon() * std::max({static_cast<T>(1), std::abs(x), std::abs(y)}) * ulp
            || std::abs(x-y) < std::numeric_limits<T>::min();  // "subnormal" case
}


/**
 * Converts numbers to strings.
 * Use std::to_string() in C++11
 * @param number Some number.
 * @return The string.
 */
template <typename T>
std::string numberToString(T number);


/**
 * Converts string to number (if possible).
 * Use std::to_string() in C++11.
 * @param text The string.
 * @return The parsed number.
 */
template <typename T>
T stringToNumber(const std::string &text);



template <class T>
std::string toString(const T & value)
{
    std::ostringstream oss;
    oss << value;
    return oss.str();
}


/**
 * Converts angle in Degree to Radian.
 * @param deg Angle in Degree.
 * @return Angle in Radian.
 */
template <typename T>
T degToRad(T degree)
{
    return degree * M_PI / 180.0;
}

/**
 * Converts angle in Radian to Degree.
 * @param rad Angle in Radian.
 * @return Angle in Degrees.
 */
template <typename T>
T radToDeg(T radian)
{
    return radian * 180.0 / M_PI;
}



/*
 * Determines the shortest distance on a quotient ring modulo N ( i.e. angular distance on a circle ).
 * @example shortestAngle(45.0, -45.0, 360.0) = -90.0
 * @param start The lower limit.
 * @param end The upper limit.
 * @param fullTurn The divisor N in (start, end) modulo N.
 * @return The shortest angle.
 */
template <typename T>
T shortestAngle( T start, T end, T fullTurn=360.0 ) {
    return fmod( ( fmod( (end - start), fullTurn) + 1.5*fullTurn), fullTurn) - 0.5*fullTurn;
}


/*
 * Clamps a number to the range [lower, upper].
 * @param value The value to be clipped.
 * @param lower The lower limit.
 * @param upper The upper limit.
 * @return The clamped value.
 */
template <typename T>
T clamp(const T& value, const T& lower, const T& upper) {
    return std::max(lower, std::min(value, upper));
}

/*
 * Fast integer power function aka. "exponentiation by squaring".
 * @note Assumes exp >= 0.
 * @param base The exponential base.
 * @param base The exponent.
 * @return The integer result of base raised to the power exp.
 */
inline int ipow(int base, int exp)
{
    int result = 1;
    while (exp > 0)
    {
        if (exp & 1)
        {
            result *= base;
        }
        exp >>= 1;
        base *= base;
    }
    return result;
}



/*
 * Argsort
 * Returns indices that would sort input vector.
 * Author: Lukasz Wiklendt, https://stackoverflow.com/questions/1577475/
 * @note Assumes template type T provides comparison operators
 * @param v Input vector to be sorted
 * @return Vector of indices to input array sorted by value
 * TODO: Provide option for custom comparators
 */
template <typename T>
std::vector<std::size_t> argsort(const std::vector<T> &v, bool ascending=true) {

    // Initialize original index locations
    std::vector<std::size_t> idx(v.size());
    std::iota(idx.begin(), idx.end(), 0);

    // Sort indices based on comparing values in vector v
    if (ascending)
    {
        std::sort(idx.begin(), idx.end(), [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});
    }
    else
    {
        std::sort(idx.begin(), idx.end(), [&v](size_t i1, size_t i2) {return v[i1] > v[i2];});
    }
    return idx;
}

/*
 * Linearly maps value in range [low1, high1] to corresponding value in range [low2, high2]
 * @example s=1/3 in the interval [0, 1] mapped to the interval [-3, 6] is 0
 *          i.e. mapLinearInterval( 1.0/3, 0.0, 1.0, -3.0, 6.0 ) = 0.0
 * @param value The value in range [low1, high1].
 * @param low1 The lower limit of the source interval.
 * @param high1 The upper limit of the source interval.
 * @param low2 The lower limit of the destination interval.
 * @param high2 The upper limit of the destination interval.
 * @return The mapped value in the destination interval.
 */
template<typename T>
T mapLinearInterval( T value, T low1, T high1, T low2, T high2 )
{
    T t = (value-low1) / (high1-low1) * (high2 - low2) + low2;
    return t;
}



/*
 * Returns iterator to closest element in SORTED container in logarithmic time.
 * @param begin Iterator pointing to first value of sorted container.
 * @param end Iterator pointing to last value of sorted container.
 * @param value Value to look for.
 * @return Iterator to closest value. Get its position with: index = result - begin.
 */
template <typename Iterator, typename T>
Iterator findNearestSorted(Iterator begin, Iterator end, T value)
{
    Iterator result = std::upper_bound(begin, end, value); // upper_bound returns the first value that compares strictly greater
    if( result != begin )
    {
        Iterator lower_result = result;
        --lower_result;
        if( result == end || ((value - *lower_result) < (*result - value)) )
        {
            result = lower_result;
        }
    }
    return result;
}

/* Returns iterator to closest element in (possibly unsorted) container in linear time
 * @param begin Iterator pointing to first value of (possibly unsorted) container.
 * @param end Iterator pointing to last value of (possibly unsorted) container.
 * @param value Value to look for.
 * @return Iterator to closest value. Get its position with: index = result - begin.
 */
template<typename InputIterator, typename T>
InputIterator findNearest(InputIterator begin, InputIterator end, T value)
{
    return std::min_element( begin, end, [&](T x, T y )
            {
                return std::abs(x - value) < std::abs(y - value);
            });
}



/*
 * Returns evenly spaced numbers over a specified, half-open or closed interval (numpy style).
 * @details Uses multiplication of stepsize to avoid additive errors.
 * @param start_ The lower limit of the desired interval (including start).
 * @param stop_ The upper limit of the desired interval.
 * @param num The number of points in the interval.
 * @param endpoint Defines if upper limit is part of the interval.
 * @return A vector containing the evenly spaced numbers.
 */
template<typename T>
std::vector<T> linspace(const T& start_, const T& stop_, const int& num, bool endpoint = true)
{
    double start = static_cast<double>( start_ );
    double stop = static_cast<double>( stop_ );
    double div = endpoint ? num - 1 : num;

    std::vector<T> result(num);
    for( int i = 0; i < num; i++ )
    {
        result[i] = static_cast<T>(i);
    }

    double delta = stop - start;

    if( num > 1) 
    {
        double step = delta / div;

        if( almostEqual(step, 0.0) )
        {
            for( int i = 0; i < num; i++ ) // Special handling for denormal numbers
            {
                result[i] /= div;
                result[i] *= delta;
            }
        }
        else
        {
            for( int i = 0; i < num; i++ )
            {
                result[i] *= step;
            }
        }
    }
    else 
    {
        // undefined step size
        for( int i = 0; i < num; i++ )
        {
            result[i] *= delta;
        }
    }

    for( int i = 0; i < num; i++ )
    {
        result[i] = static_cast<T>( result[i] + start );
    }

    if( endpoint && num > 1)
    {
        if (!result.empty()) {
            result.back() = stop;
        }
    }

    return result;
}


// Draws samples from a uniform distribution in range [low, high] if inclusive is true, else [low, high)
template<typename T>
std::vector<T> random_uniform(const T low, const T high, std::size_t size, bool inclusive = true, bool fixedSeed = true)
{
    static_assert(std::is_integral<T>::value || std::is_floating_point<T>::value, "Random uniform distribution is defined for integer and floating point types only!");
    
    std::vector<T> vec(size);
    
    // Conditional template type (C++, I hate it with a passion!)
    using uniform_distribution = 
        typename std::conditional<
        std::is_integral<T>::value,
        std::uniform_int_distribution<T>,
        std::uniform_real_distribution<T> >::type;
        
    uniform_distribution distribution;
    
    // Initialize random number generator
    std::mt19937 generator; // 32-bit Mersenne Twister (Matsumoto and Nishimura, 1998)
    if (fixedSeed)
    {
        generator.seed(42); // Fixed seed
    }
    else
    {
        std::random_device device; // WARNING: No entropy guarantees
        generator.seed(device()); // 32 bit seed from operating system 
    }
    
    // Integers
    if ( std::is_integral<T>::value )
    {
        if (inclusive)
        {
            distribution = uniform_distribution(low, high); // [a, b]
        }
        else
        {
            distribution = uniform_distribution(low, high-1); // [a, b]
        }
        
        for( std::size_t i = 0; i < size; i++ )
        {
            vec[i] = distribution(generator);
        }
    }
    
    // Floating point
    else if ( std::is_floating_point<T>::value )
    {
        if (inclusive)
        {
            distribution = uniform_distribution(low, std::nextafter(high, std::numeric_limits<T>::max())); // [a, b)
        }
        else
        {
            distribution = uniform_distribution(low, high); // [a, b)
        }
        
        for( std::size_t i = 0; i < size; i++ )
        {
            vec[i] = distribution(generator);
        }
    }
    
    return vec;
}



// Draws samples from a normal distribution
template<typename T>
std::vector<T> random_normal(const T mean, const T stddev, std::size_t size, bool fixedSeed = true)
{
    std::vector<T> vec(size);
    
    std::normal_distribution<T> distribution(mean, stddev);
   
    // Initialize random number generator
    std::mt19937 generator; // 32-bit Mersenne Twister (Matsumoto and Nishimura, 1998)
    if (fixedSeed)
    {
        generator.seed(42); // Fixed seed
    }
    else
    {
        std::random_device device; // WARNING: No entropy guarantees
        generator.seed(device()); // 32 bit seed from operating system
    }
  
    for( std::size_t i = 0; i < size; i++ )
    {
        vec[i] = distribution(generator);
    }

    return vec;
}



template <typename T>
double computeMedian(const std::vector<T>& v)
{
    std::vector<T> tmp(std::begin(v), std::end(v));
    std::size_t middle = tmp.size() / 2;

    // O(n) but worst case could be O(n log n) depending on implementation
    std::nth_element(tmp.begin(), tmp.begin() + middle, tmp.end());

    if (v.size() == 0)
    {
        return std::numeric_limits<double>::quiet_NaN();
    }
    else if(tmp.size() % 2 != 0) 
    {
        return tmp.at(middle); // Odd
    }
    else
    {
        // Even: Halfway point between the two middle values
        auto max_it = std::max_element(tmp.begin(), tmp.begin() + middle); // O(n)
        return (*max_it + tmp.at(middle)) / 2.0;
    }
}

// ~1.25x speedup over non-destructive version
template <typename T>
double computeMedian_inplace(std::vector<T> &v)
{
    std::size_t middle = v.size() / 2;
    
    // O(n) but worst case could be O(n log n) depending on implementation
    std::nth_element(v.begin(), v.begin() + middle, v.end());
    
    if (v.size() == 0)
    {
        return std::numeric_limits<double>::quiet_NaN();
    }
    else if (v.size() % 2 != 0)
    {
        return v[middle]; // Odd
    }
    else
    {
        // Even: Halfway point between the two middle values
        auto max_it = std::max_element(v.begin(), v.begin() + middle); // O(n)
        return(*max_it + v[middle]) / 2.0;
    }
}


// May suffer from floating point rounding errors during accumulation (only relevant for large sample size).
// Use Kahan summation algorithm for better precision.
// https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance also lists online and weighted variants.
template <typename T>
std::pair<double, double> computeVariance(const std::vector<T>& v, const int dof=0)
{
    std::size_t n = v.size();

    // Mean
    T mean = std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(n);

    // Mean squared deviation from the mean aka. variance ;-)
    auto variance_func = [&mean](T accumulator, const T& x)
    {
        return accumulator + (x - mean) * (x - mean);
    };

    double variance = std::accumulate(v.begin(), v.end(), 0.0, variance_func) / static_cast<double>(n - dof); // Bessel's correction
    
    return std::make_pair(static_cast<double>(mean), variance);
}


template <typename T>
std::pair<double, double> computeStandardDeviation(const std::vector<T>& v, const int dof=0)
{
    double mean, variance;
    std::tie(mean, variance) = computeVariance(v, dof);
    return std::make_pair(mean, std::sqrt(variance));
}


// Median absolute deviation (MAD)
// In the presence of outliers, median and MAD are often used instead of mean and standard deviation.
// - The MAD "has emerged as the single most useful ancillary estimate of scale" Huber (1981).
// - High breakdown value (50% outliers).
// - Inefficient for Normal Distributions (37% efficiency).
// - Computes symmetric statistic about a location estimate, thus the MAD is not suitable for skewed distributions.
// - Influence function has discontinuities.
// - k = 1.0 / ppf(3/4.0) # ~1.4826, ppf refers to the quantile or percent point function (inverse of cdf) of the Normal Distribution
//
// History lesson:
// First mentioned by Gauss: "Bestimmung der Genauigkeit der Beobachtungen", Carl Friedrich Gauss (1816)
// Rediscovered by Hampel: "The Influence Curve and its Role in Robust Estimation", Frank Hampel (1974)
// Popularized by Huber: "Robust statistics", Peter J. Huber (1981)
template <typename T>
std::pair<double, double> computeMedianAbsoluteDeviation(const std::vector<T>& v, const double k=1.482602218505602)
{
    std::vector<T> tmp = v; // Copy
    T median = computeMedian_inplace(tmp);
     
    for(std::size_t i = 0; i < tmp.size(); i++)
    {
        tmp[i] = std::fabs(tmp[i] - median);
    }
    double mad = k * computeMedian_inplace(tmp); // k is a correction factor for unbiased consistency w.r.t. reference distribution

    return std::make_pair(static_cast<double>(median), mad);
}



static inline void convertToGray(const cv::Mat& src, cv::Mat& dst)
{
    int numChannels = src.channels();
    if (numChannels == 1) {
        src.copyTo(dst); // Avoids copy if possible (benchmarked)
    }
    else if (numChannels == 3) {
        cv::cvtColor(src, dst, cv::COLOR_BGR2GRAY);
    }
    else if (numChannels == 4) {
        cv::cvtColor(src, dst, cv::COLOR_BGRA2GRAY);
    }
}

static inline void convertToBGR(const cv::Mat& src, cv::Mat& dst)
{
    int numChannels = src.channels();
    if (numChannels == 1) {
        cv::cvtColor(src, dst, cv::COLOR_GRAY2BGR);
    }
    else if (numChannels == 3) {
        src.copyTo(dst); // Avoids copy if possible (benchmarked)
    }
    else if (numChannels == 4) {
        cv::cvtColor(src, dst, cv::COLOR_BGRA2BGR);
    }
}

static inline void convertToBGRA(const cv::Mat& src, cv::Mat& dst)
{
    int numChannels = src.channels();
    if (numChannels == 1) {
        cv::cvtColor(src, dst, cv::COLOR_GRAY2BGRA);
    }
    else if (numChannels == 3) {
        cv::cvtColor(src, dst, cv::COLOR_BGR2BGRA);
    }
    else if (numChannels == 4) {
        src.copyTo(dst); // Avoids copy if possible (benchmarked)
    }
}


//------------------------------
// Optimal Gaussian kernel size
//------------------------------
// For your interest: Two related techniques to compute the optimal kernel size
// 1) Simple definition based on Full Width at Half Maximum (FWHM).
//    Reference: https://en.wikipedia.org/wiki/Full_width_at_half_maximum
//    diameter = 2*radius + 1
//    sigma = diameter / 2.35482004503 // 2*sqrt(2*ln(2))
//    radius = (2.35482004503 * sigma - 1) / 2
//    Derived from FWHM = 2*sqrt(2*ln(2))*sigma
//
// 2) Cutoff equation w.r.t. acceptable quantization error.
//    Reference: https://patrickfuller.github.io/gaussian-blur-image-processing-for-scientists-and-engineers-part-4
//    radius = sqrt(-2*sigma^2 * ln(p))
//    sigma = radius / sqrt(-ln(p))
//    Derived from p = g(r) / g(0) where p is the cutoff percentage, g(x) is the Gaussian function and r is the kernel radius.
//    Note: An acceptable quantization error of 0.5 is equivalent to one FWHM.

// Computes kernel radius as a function of the standard deviation.
// sigma: Standard deviation of the Gaussian
// cutoff_percentage: Acceptable quantization error.
static inline int optimalRadius(double sigma=1.0, double cutoff_percentage=0.05)
{
    double radius = sigma * std::sqrt(-2*std::log(cutoff_percentage));
    int radiusInteger = std::max(cvRound(radius), 1); // Round half to even
    return radiusInteger;
}

// Computes standard deviation as a function of the kernel radius.
// radius: Half-size of Gaussian kernel. Use odd kernel diameter of size 2 * radius + 1.
// cutoff_percentage: Acceptable quantization error.
static inline double optimalSigma(int radius=3, double cutoff_percentage=0.05)
{ 
    double sigma = radius / std::sqrt(-2*std::log(cutoff_percentage));
    return sigma;
}


} // namespace utils

#endif /* ARTSYSTEM_SRC_TOOLS_POSECNN_UTILS_H_ */
