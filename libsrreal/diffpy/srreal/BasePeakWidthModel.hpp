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
* class PeakWidthModel -- base class for calculation of peak widths.
*     The The calculate function takes a BondGenerator instance and
*     returns full width at half maximum, based on peak model parameters
*     and anisotropic displacement parameters of atoms in the pair.
*
* $Id$
*
*****************************************************************************/

#ifndef BASEPEAKWIDTHMODEL_HPP_INCLUDED
#define BASEPEAKWIDTHMODEL_HPP_INCLUDED

#include <string>
#include <map>

namespace diffpy {
namespace srreal {

class BaseBondGenerator;

class BasePeakWidthModel
{
    public:

        // constructors
        virtual BasePeakWidthModel* create() const = 0;
        virtual BasePeakWidthModel* copy() const = 0;
        virtual ~BasePeakWidthModel()  { }

        // methods
        virtual const std::string& type() const = 0;
        virtual double calculate(const BaseBondGenerator&) const = 0;

        // comparison with derived classes
        virtual bool operator==(const BasePeakWidthModel&) const = 0;

    private:

        // class method for registration
        friend BasePeakWidthModel* createPeakWidthModel(const std::string&);
        friend bool registerPeakWidthModel(const BasePeakWidthModel&);
        static std::map<std::string, BasePeakWidthModel*>& getRegistry();
};

// Factory functions for Peak Width Models -----------------------------------

BasePeakWidthModel* createPeakWidthModel(const std::string& tp);
bool registerPeakWidthModel(const BasePeakWidthModel&);

}   // namespace srreal
}   // namespace diffpy

// Implementation ------------------------------------------------------------

#include <diffpy/srreal/BasePeakWidthModel.ipp>

#endif  // BASEPEAKWIDTHMODEL_HPP_INCLUDED
