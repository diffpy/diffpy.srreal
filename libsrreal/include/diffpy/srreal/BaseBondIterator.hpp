/*****************************************************************************
*
* diffpy.srreal     by DANSE Diffraction group
*                   Simon J. L. Billinge
*                   (c) 2009 Trustees of the Columbia University
*                   in the City of New York.  All rights reserved.
*
* File coded by:    Pavol Juhas
*
* See AUTHORS.txt for a list of people who contributed.
* See LICENSE.txt for license information.
*
******************************************************************************
*
* class BaseBondIterator -- semi-abstract class for an iterator
*     over all atom pairs containing specified anchor atom.
*
* $Id$
*
*****************************************************************************/

#ifndef BASEBONDITERATOR_HPP_INCLUDED
#define BASEBONDITERATOR_HPP_INCLUDED

#include "R3linalg.hpp"

namespace diffpy {
namespace srreal {

class StructureAdapter;

class BaseBondIterator
{
    public:

        // constructor
        BaseBondIterator(const StructureAdapter*);

        // methods
        // loop control
        void rewind();
        bool finished() const;
        void next();

        // configuration
        void selectAnchorSite(int);
        void selectSiteRange(int first, int last);
        void includeSelfPairs(bool);

        // get data
        virtual const R3::Vector& r0() const = 0;
        virtual const R3::Vector& r1() const = 0;
        double distance() const;

    protected:

        // data
        int msite_anchor;
        int msite_first;
        int msite_last;
        int msite_current;
        bool minclude_self_pairs;
        const StructureAdapter* mstructure;

        // bond data
        R3::Vector mbond_r0;
        R3::Vector mbond_r1;

        // methods
        virtual bool iterateSymmetry();

    private:

        // methods
        void skipSelfPair();
        void setFinishedFlag();

};


}   // namespace srreal
}   // namespace diffpy

#endif  // BASEBONDITERATOR_HPP_INCLUDED
