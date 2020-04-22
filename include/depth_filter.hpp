/**
* @file depth_filter.hpp
* @brief gaussian distribution representation for depth
*
* @date 22.04.2020
* @author Amin Abouee
*
* @section DESCRIPTION
*
*
*/
#ifndef __DEPTH_FILTER_H__
#define __DEPTH_FILTER_H__

#include <iostream>

class DepthFilter final
{
public:
    //C'tor
    explicit DepthFilter();
    //Copy C'tor
    DepthFilter(const DepthFilter& rhs) = delete;
    //move C'tor
    DepthFilter(DepthFilter&& rhs) = delete;
    //Copy assignment operator
    DepthFilter& operator=(const DepthFilter& rhs) = delete;
    //move assignment operator
    DepthFilter& operator=(DepthFilter&& rhs) = delete;
    //D'tor
    ~DepthFilter() = default;

 private:

 };

 #endif /* __DEPTH_FILTER_H__ */