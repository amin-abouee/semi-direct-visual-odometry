#include "utils.hpp"

Eigen::IOFormat utils::eigenFormat()
{
    return Eigen::IOFormat( 6, Eigen::DontAlignCols, ", ", " , ", "[", "]", "[", "]" );
}