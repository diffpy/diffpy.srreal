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
* Various common routines useful for PDF calculation:
*     meanSquareDisplacement
*     bandPassFilter
*
* $Id$
*
*****************************************************************************/

#ifndef PDFUTILS_HPP_INCLUDED
#define PDFUTILS_HPP_INCLUDED

#include <diffpy/srreal/R3linalg.hpp>

namespace diffpy {
namespace srreal {

// Calculate MSD along specified direction in Cartesian space.
double meanSquareDisplacement(const R3::Matrix& Uijcartn, const R3::Vector& s,
        bool anisotropy=false);

}   // namespace srreal
}   // namespace diffpy

#endif  // PDFUTILS_HPP_INCLUDED
