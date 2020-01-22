#ifndef __SYSTEM_HPP__
#define __SYSTEM_HPP__

#include <iostream>
#include <Eigen/Core>

class System final
{
public:
    explicit System();
    System (const System& rhs) = delete;
    System (System&& rhs) = delete;
    System& operator= (const System& rhs) = delete;
    System& operator= (System&& rhs) = delete;
    ~System() = default;

private:
};

#endif /* __SYSTEM_HPP__ */