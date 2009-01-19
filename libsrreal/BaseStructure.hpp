#ifndef BASESTRUCTURE_HPP_INCLUDED
#define BASESTRUCTURE_HPP_INCLUDED

namespace diffpy {

class BaseBondIterator;


class BaseStructure
{
    public:

        // methods
        virtual int countSites() const = 0;
        virtual BaseBondIterator* createBondIterator() const = 0;

};


}   // namespace diffpy

#endif  // BASESTRUCTURE_HPP_INCLUDED
