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
* class ZeroBaseline -- linear PDF baseline
*
* $Id$
*
*****************************************************************************/

#include <diffpy/srreal/ZeroBaseline.hpp>

using namespace std;

namespace diffpy {
namespace srreal {

// Constructors --------------------------------------------------------------

PDFBaseline* ZeroBaseline::create() const
{
    PDFBaseline* rv = new ZeroBaseline();
    return rv;
}


PDFBaseline* ZeroBaseline::copy() const
{
    PDFBaseline* rv = new ZeroBaseline(*this);
    return rv;
}

// Public Methods ------------------------------------------------------------

const string& ZeroBaseline::type() const
{
    static string rv = "zero";
    return rv;
}


double ZeroBaseline::operator()(const double& r) const
{
    return 0.0;
}

// Registration --------------------------------------------------------------

bool reg_ZeroBaseline = registerPDFBaseline(ZeroBaseline());

}   // namespace srreal
}   // namespace diffpy

// End of file
