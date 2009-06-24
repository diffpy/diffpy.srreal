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
* class LinearBaseline -- linear PDF baseline
*
* $Id$
*
*****************************************************************************/

#include <diffpy/srreal/LinearBaseline.hpp>

using namespace std;

namespace diffpy {
namespace srreal {

// Constructors --------------------------------------------------------------

LinearBaseline::LinearBaseline()
{
    this->setSlope(0.0);
    this->registerDoubleAttribute("slope", this,
            &LinearBaseline::getSlope,
            &LinearBaseline::setSlope);
}


PDFBaseline* LinearBaseline::create() const
{
    PDFBaseline* rv = new LinearBaseline();
    return rv;
}


PDFBaseline* LinearBaseline::copy() const
{
    PDFBaseline* rv = new LinearBaseline(*this);
    return rv;
}

// Public Methods ------------------------------------------------------------

const string& LinearBaseline::type() const
{
    static string rv = "linear";
    return rv;
}


double LinearBaseline::operator()(const double& r) const
{
    return (this->getSlope() * r);
}


void LinearBaseline::setSlope(double sc)
{
    mslope = sc;
}


const double& LinearBaseline::getSlope() const
{
    return mslope;
}


// Registration --------------------------------------------------------------

bool reg_LinearBaseline = registerPDFBaseline(LinearBaseline());

}   // namespace srreal
}   // namespace diffpy

// End of file
