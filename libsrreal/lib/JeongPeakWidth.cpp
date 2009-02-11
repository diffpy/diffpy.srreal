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
* class JeongPeakWidth -- peak width model based on FIXME paper by Jeong
*
* $Id$
*
*****************************************************************************/

#include <diffpy/srreal/JeongPeakWidth.hpp>
#include <diffpy/srreal/BaseBondGenerator.hpp>

using namespace std;
using namespace diffpy::srreal;

// Constructors --------------------------------------------------------------

JeongPeakWidth::JeongPeakWidth()
{
    this->setDelta1(0.0);
    this->setDelta2(0.0);
    this->setQbroad(0.0);
}


BasePeakWidthModel* JeongPeakWidth::create() const
{
    BasePeakWidthModel* rv = new JeongPeakWidth();
    return rv;
}


BasePeakWidthModel* JeongPeakWidth::copy() const
{
    BasePeakWidthModel* rv = new JeongPeakWidth(*this);
    return rv;
}

// Public Methods ------------------------------------------------------------

bool JeongPeakWidth::operator==(const BasePeakWidthModel& other) const
{
    if (this == &other)  return true;
    const JeongPeakWidth* jpwm =
        dynamic_cast<const JeongPeakWidth*>(&other);
    bool rv = jpwm &&
        mdelta1 == jpwm->mdelta1 &&
        mdelta2 == jpwm->mdelta2 &&
        mqbroad == jpwm->mqbroad;
    return rv;
}


const string& JeongPeakWidth::type() const
{
    static const string rv = "jeong";
    return rv;
}


double JeongPeakWidth::calculate(const BaseBondGenerator& bnds) const
{
    double dw_fwhm = this->DebyeWallerPeakWidth::calculate(bnds);
    double r = bnds.distance();
    double corr = 1.0 - this->getDelta1()/r - this->getDelta2()/pow(r, 2) +
        pow(this->getQbroad()*r, 2);
    double fwhm = (corr > 0) ? (sqrt(corr) * dw_fwhm) : dw_fwhm;
    return fwhm;
}


const double& JeongPeakWidth::getDelta1() const
{
    return mdelta1;
}


void JeongPeakWidth::setDelta1(double delta1)
{
    mdelta1 = delta1;
}


const double& JeongPeakWidth::getDelta2() const
{
    return mdelta2;
}


void JeongPeakWidth::setDelta2(double delta2)
{
    mdelta2 = delta2;
}


const double& JeongPeakWidth::getQbroad() const
{
    return mqbroad;
}


void JeongPeakWidth::setQbroad(double qbroad)
{
    mqbroad = qbroad;
}

// Registration --------------------------------------------------------------

bool reg_JeongPeakWidthModel = registerPeakWidthModel(JeongPeakWidth());

// End of file
