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
* class StepCutEnvelope -- empirical step-function PDF envelope. 
*
* $Id$
*
*****************************************************************************/

#include <diffpy/srreal/StepCutEnvelope.hpp>

using namespace std;

namespace diffpy {
namespace srreal {

// Constructor ---------------------------------------------------------------

StepCutEnvelope::StepCutEnvelope()
{
    this->setStepCut(0.0);
    this->registerDoubleAttribute("stepcut", this,
            &StepCutEnvelope::getStepCut, &StepCutEnvelope::setStepCut);
}


PDFEnvelope* StepCutEnvelope::create() const
{
    PDFEnvelope* rv = new StepCutEnvelope();
    return rv;
}


PDFEnvelope* StepCutEnvelope::copy() const
{
    PDFEnvelope* rv = new StepCutEnvelope(*this);
    return rv;
}

// Public Methods ------------------------------------------------------------

const string& StepCutEnvelope::type() const
{
    static string rv = "stepcut";
    return rv;
}


double StepCutEnvelope::operator()(const double& r) const
{
    double rv = (mstepcut > 0.0 && r > mstepcut) ? 0.0 : 1.0;
    return rv;
}


void StepCutEnvelope::setStepCut(double sc)
{
    mstepcut = sc;
}


const double& StepCutEnvelope::getStepCut() const
{
    return mstepcut;
}

// Registration --------------------------------------------------------------

bool reg_StepCutEnvelope = registerPDFEnvelope(StepCutEnvelope());

}   // namespace srreal
}   // namespace diffpy

// End of file
