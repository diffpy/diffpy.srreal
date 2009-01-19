#ifndef VR3STRUCTURE_HPP_INCLUDED
#define VR3STRUCTURE_HPP_INCLUDED

#include <vector>
#include "R3linalg.hpp"
#include "BaseStructure.hpp"
#include "BaseBondIterator.hpp"

namespace diffpy {

typedef std::vector<R3::Vector> VR3Structure;

class VR3Adaptor : public BaseStructure
{
    public:

        class VR3BondIterator;
        // constructors
        VR3Adaptor(const VR3Structure& vr3s)
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

                VR3BondIterator(const VR3Adaptor* stru) : BaseBondIterator(stru)
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


inline BaseStructure* createPQAdaptor(const VR3Structure& vr3stru)
{
    BaseStructure* adaptor = new VR3Adaptor(vr3stru);
    return adaptor;
}


}   // namespace diffpy

#endif  // VR3STRUCTURE_HPP_INCLUDED
