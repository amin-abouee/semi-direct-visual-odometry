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
#include <memory>

#include "frame.hpp"

namespace Matcher
{
    void computeOpticalFlowSparse(std::shared_ptr<Frame>& refFrame, std::shared_ptr<Frame>& curFrame, const uint32_t patchSize);

    void computeEssentialMatrix(std::shared_ptr<Frame>& refFrame, std::shared_ptr<Frame>& curFrame, const double reproError, Eigen::Matrix3d& E);

    void templateMatching( const std::shared_ptr<Frame>& refFrame, std::shared_ptr<Frame>& curFrame, const uint16_t patchSzRef, const uint16_t patchSzCur );
// private:
}

#endif /* __MATCHER_H__ */