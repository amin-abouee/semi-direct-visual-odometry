#ifndef __TRIANGULATION_HPP__
#define __TRIANGULATION_HPP__

#include <iostream>

class Triangulation final
{
public:
    explicit Triangulation();
    Triangulation(const Triangulation& rhs) = delete;
    Triangulation(Triangulation&& rhs) = delete;
    Triangulation& operator = (const Triangulation& rhs) = delete;
    Triangulation& operator = (Triangulation&& rhs) = delete;
    ~Triangulation() = default;

    triangulatePointDLT();

private:
};
#endif /* __TRIANGULATION_HPP__ */