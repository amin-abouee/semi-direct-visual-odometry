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
#include "frame.hpp"

// class Matcher final
// {
// public:
//     // C'tor
//     explicit Matcher() = default;
//     // Copy C'tor
//     Matcher( const Matcher& rhs ) = default;
//     // move C'tor
//     Matcher( Matcher&& rhs ) = default;
//     // Copy assignment operator
//     Matcher& operator=( const Matcher& rhs ) = default;
//     // move assignment operator
//     Matcher& operator=( Matcher&& rhs ) = default;
//     // D'tor
//     ~Matcher() = default;
namespace Matcher
{
    void computeOpticalFlowSparse(std::shared_ptr<Frame>& refFrame, std::shared_ptr<Frame>& curFrame, const uint16_t patchSize);

    void computeEssentialMatrix(std::shared_ptr<Frame>& refFrame, std::shared_ptr<Frame>& curFrame, const double reproError, Eigen::Matrix3d& E);

    // bool findEpipolarMatch( std::shared_ptr<Frame>& refFrame,
    //                         std::shared_ptr<Frame>& curFrame,
    //                         Feature& ft,
    //                         const double minDepth,
    //                         const double maxDepth,
    //                         double& estimatedDepth );

    void templateMatching( const std::shared_ptr<Frame>& refFrame, std::shared_ptr<Frame>& curFrame, const uint16_t patchSzRef, const uint16_t patchSzCur );
// private:
}

#endif /* __MATCHER_H__ */