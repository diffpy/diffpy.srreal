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
* class BaseBondGenerator -- semi-abstract class for a generation
*     of all atom pairs containing specified anchor atom.
*
* $Id$
*
*****************************************************************************/

#ifndef BASEBONDGENERATOR_HPP_INCLUDED
#define BASEBONDGENERATOR_HPP_INCLUDED

#include "R3linalg.hpp"

namespace diffpy {
namespace srreal {

class StructureAdapter;

class BaseBondGenerator
{
    public:

        // constructor
        BaseBondGenerator(const StructureAdapter*);
        virtual ~BaseBondGenerator()  { }

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
        virtual const R3::Vector& r0() const;
        virtual const R3::Vector& r1() const;
        double distance() const;
        const R3::Vector& r01() const;
        virtual double msd0() const;
        virtual double msd1() const;
        double msd() const;

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

#endif  // BASEBONDGENERATOR_HPP_INCLUDED
