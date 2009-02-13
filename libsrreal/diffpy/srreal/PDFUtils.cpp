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

#include <diffpy/srreal/PDFUtils.hpp>

namespace diffpy {
namespace srreal {


double meanSquareDisplacement(const R3::Matrix& Uijcartn,
        const R3::Vector& s, bool anisotropy)
{
    double rv;
    if (anisotropy)
    {
        assert(R3::norm(s) > 0);
        assert(Uijcartn(0,1) == Uijcartn(1,0));
        assert(Uijcartn(0,2) == Uijcartn(2,0));
        assert(Uijcartn(1,2) == Uijcartn(2,1));
        static R3::Vector sn;
        sn = s / R3::norm(s);
        rv = Uijcartn(0,0) * sn(0) * sn(0) +
             Uijcartn(1,1) * sn(1) * sn(1) +
             Uijcartn(2,2) * sn(2) * sn(2) +
             2 * Uijcartn(0,1) * sn(0) * sn(1) +
             2 * Uijcartn(0,2) * sn(0) * sn(2) +
             2 * Uijcartn(1,2) * sn(1) * sn(2);
    }
    else
    {
        assert(Uijcartn(0,0) == Uijcartn(1,1));
        assert(Uijcartn(0,0) == Uijcartn(2,2));
        rv = Uijcartn(0,0);
    }
    return rv;
}


}   // namespace srreal
}   // namespace diffpy

// End of file
