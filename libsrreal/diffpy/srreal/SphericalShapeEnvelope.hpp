/*****************************************************************************
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
* class SphericalShapeEnvelope -- PDF envelope due to spherical shape factor
*
* $Id$
*
*****************************************************************************/

#ifndef SPHERICALSHAPEENVELOPE_HPP_INCLUDED
#define SPHERICALSHAPEENVELOPE_HPP_INCLUDED

#include <diffpy/srreal/PDFEnvelope.hpp>

namespace diffpy {
namespace srreal {

/// @class SphericalShapeEnvelope
/// @brief PDF decay due to finite spherical shape of the sample.
/// Spherical shape envelope is not applied for spherical particle
/// diameter, spdiameter, smaller or equal to zero.

class SphericalShapeEnvelope : public PDFEnvelope
{
    public:

        // constructors
        SphericalShapeEnvelope();
        virtual PDFEnvelope* create() const;
        virtual PDFEnvelope* copy() const;

        // methods

        virtual const std::string& type() const;
        virtual double operator()(const double& r) const;
        void setSPDiameter(double spd);
        const double& getSPDiameter() const;

    private:

        // data
        double mspdiameter;

};  // class SphericalShapeEnvelope

}   // namespace srreal
}   // namespace diffpy

#endif  // SPHERICALSHAPEENVELOPE_HPP_INCLUDED
