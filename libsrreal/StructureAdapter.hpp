#ifndef STRUCTUREADAPTER_HPP_INCLUDED
#define STRUCTUREADAPTER_HPP_INCLUDED

namespace diffpy {

class BaseBondIterator;


class StructureAdapter
{
    public:

        // methods
        virtual int countSites() const = 0;
        virtual BaseBondIterator* createBondIterator() const = 0;

};


}   // namespace diffpy

#endif  // STRUCTUREADAPTER_HPP_INCLUDED
