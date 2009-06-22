/*****************************************************************************
*
* diffpy.srreal     by DANSE Diffraction group
*                   Simon J. L. Billinge
*                   (c) 2009 Trustees of the Columbia University
*                   in the City of New York.  All rights reserved.
*
* File coded by:    Christopher Farrow, Pavol Juhas
*
* See AUTHORS.txt for a list of people who contributed.
* See LICENSE.txt for license information.
*
******************************************************************************
*
* class DebyeWallerPeakWidth -- peak width calculated assuming independent
*     thermal vibrations of atoms forming a pair.
*
* $Id$
*
*****************************************************************************/

#ifndef DEBYEWALLERPEAKWIDTH_HPP_INCLUDED
#define DEBYEWALLERPEAKWIDTH_HPP_INCLUDED

#include <diffpy/srreal/PeakWidthModel.hpp>

namespace diffpy {
namespace srreal {


class DebyeWallerPeakWidth : public PeakWidthModel
{
    public:

        // constructors
        virtual PeakWidthModel* create() const;
        virtual PeakWidthModel* copy() const;

        // comparison with derived classes
        virtual bool operator==(const PeakWidthModel&) const;

        // methods
        virtual const std::string& type() const;
        virtual double calculate(const BaseBondGenerator&) const;
        virtual double calculateFromMSD(double msdval) const;

};


}   // namespace srreal
}   // namespace diffpy

#endif  // DEBYEWALLERPEAKWIDTH_HPP_INCLUDED
