#ifndef BASESTRUCTURE_HPP_INCLUDED
#define BASESTRUCTURE_HPP_INCLUDED

namespace diffpy {

class BaseBondIterator;
class BaseBondPair;


class BaseStructure
{
    public:

        // methods
        int countSites() const;
        BaseBondIterator* createBondIterator() const;
        BaseBondPair* createBondPair() const;

};


}   // namespace diffpy

#endif  // BASESTRUCTURE_HPP_INCLUDED
