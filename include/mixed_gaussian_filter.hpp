/**
* @file mixed_gaussian_filter.hpp
* @brief mixed representation of depth based on gaussian adn beta distribution
*
* @date 11.05.2020
* @author Amin Abouee
*
* @section DESCRIPTION
*
*
*/
#ifndef __MIXED_GAUSSIAN_FILTER_H__
#define __MIXED_GAUSSIAN_FILTER_H__

#include <iostream>

class MixedGaussianFilter final
{
public:
    //C'tor
    explicit MixedGaussianFilter();
    //Copy C'tor
    MixedGaussianFilter(const MixedGaussianFilter& rhs) = delete;
    //move C'tor
    MixedGaussianFilter(MixedGaussianFilter&& rhs) = delete;
    //Copy assignment operator
    MixedGaussianFilter& operator=(const MixedGaussianFilter& rhs) = delete;
    //move assignment operator
    MixedGaussianFilter& operator=(MixedGaussianFilter&& rhs) = delete;
    //D'tor
    ~MixedGaussianFilter() = default;

 private:

 };

 #endif /* __MIXED_GAUSSIAN_FILTER_H__ */