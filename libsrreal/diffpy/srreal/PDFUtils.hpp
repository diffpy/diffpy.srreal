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

#include <valarray>
#include <diffpy/srreal/R3linalg.hpp>

namespace diffpy {
namespace srreal {

// Calculate MSD along specified direction in Cartesian space.
double meanSquareDisplacement(const R3::Matrix& Uijcartn, const R3::Vector& s,
        bool anisotropy=false);

// Apply band pass filter to a sequence of doubles
template <class Ti>
void bandPassFilter(Ti first, Ti last, double dr, double qmin, double qmax);

// Implementation of bandPassFilter using padded complex valarray
void bandPassFilterCValarray(std::valarray<double>& ycpa,
        double dr, double qmin, double qmax);

}   // namespace srreal
}   // namespace diffpy

// Implementation ------------------------------------------------------------

#include <diffpy/srreal/PDFUtils.ipp>

#endif  // PDFUTILS_HPP_INCLUDED
