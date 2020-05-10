#include "map.hpp"

#include "easylogging++.h"
#define Map_Log( LEVEL ) CLOG( LEVEL, "Map" )

Map::Map()
{

}

void Map::reset()
{
    m_keyFrames.clear();
    m_trashPoints.clear();
}

bool Map::removeFrame( std::shared_ptr< Frame >& frame )
{

}

bool Map::removePoint( std::shared_ptr< Point >& point )
{

}

bool Map::removeFeature( std::shared_ptr< Feature >& feature )
{

}

void Map::addKeyframe( std::shared_ptr< Frame >& frame )
{

}

std::shared_ptr< Frame >& Map::getCloseKeyframe (std::shared_ptr< Frame >& frame) const
{

}

void Map::getCloseKeyframes (std::shared_ptr< Frame >& frame, std::vector<keyframeDistance>& closeKeyframes) const
{

}

void Map::transform (const Eigen::Matrix3d& R, const Eigen::Vector3d& t, const double& s) const
{

}

void Map::emptyTrash()
{

}

std::shared_ptr< Frame >& Map::lastKeyframes()
{

}

std::size_t Map::sizeKeyframes () const
{

}
