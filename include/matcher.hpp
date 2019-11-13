/**
* @file matcher.hpp
* @brief matching between frames
*
* @date 13.11.2019
* @author Amin Abouee
*
* @section DESCRIPTION
*
*
*/
#ifndef __MATCHER_H__
#define __MATCHER_H__

#include <iostream>

class Matcher final
{
public:
    //C'tor
    explicit Matcher() = default;
    //Copy C'tor
    Matcher(const Matcher& rhs) = default;
    //move C'tor
    Matcher(Matcher&& rhs) = default;
    //Copy assignment operator
    Matcher& operator=(const Matcher& rhs) = default;
    //move assignment operator
    Matcher& operator=(Matcher&& rhs) = default;
    //D'tor
    ~Matcher() = default;

    bool findEpipolarMatch()

 private:

 };

 #endif /* __MATCHER_H__ */