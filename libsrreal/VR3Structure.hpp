#ifndef VR3STRUCTURE_HPP_INCLUDED
#define VR3STRUCTURE_HPP_INCLUDED

#include <vector>
#include "R3linalg.hpp"
#include "StructureAdapter.hpp"
#include "BaseBondIterator.hpp"

namespace diffpy {

typedef std::vector<R3::Vector> VR3Structure;

class VR3Adapter : public StructureAdapter
{
    public:

        class VR3BondIterator;
        // constructors
        VR3Adapter(const VR3Structure& vr3s)
        {
            mvr3structure = &vr3s;
        }

        // methods
        virtual int countSites() const
        {
            return mvr3structure->size();
        }

        virtual BaseBondIterator* createBondIterator() const
        {
            BaseBondIterator* bbi = new VR3BondIterator(this);
            return bbi;
        }

        class VR3BondIterator : public BaseBondIterator
        {
            public:

                VR3BondIterator(const VR3Adapter* stru) : BaseBondIterator(stru)
                { 
                    mvr3structure = stru->mvr3structure;
                }

                virtual const R3::Vector& r0() const
                {
                    return mvr3structure->at(msite_anchor);
                }

                virtual const R3::Vector& r1() const
                {
                    return mvr3structure->at(msite_current);
                }

            private:

                const VR3Structure* mvr3structure;
        };

    private:

        // data
        const VR3Structure* mvr3structure;

};


inline StructureAdapter* createPQAdapter(const VR3Structure& vr3stru)
{
    StructureAdapter* adapter = new VR3Adapter(vr3stru);
    return adapter;
}


}   // namespace diffpy

#endif  // VR3STRUCTURE_HPP_INCLUDED
