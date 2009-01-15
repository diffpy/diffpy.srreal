#ifndef BASEBONDPAIR_HPP_INCLUDED
#define BASEBONDPAIR_HPP_INCLUDED

#include "R3linalg.hpp"

namespace diffpy {


class BaseBondPair
{
    public:

        // methods
        const R3::Vector& r0() const;
        const R3::Vector& r1() const;
        double distance() const;

    protected:

        // data
        R3::Vector mr0;
        R3::Vector mr1;

};


}   // namespace diffpy

#endif  // BASEBONDPAIR_HPP_INCLUDED
