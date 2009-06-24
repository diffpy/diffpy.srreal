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

#ifndef STEPCUTENVELOPE_HPP_INCLUDED
#define STEPCUTENVELOPE_HPP_INCLUDED

#include <diffpy/srreal/PDFEnvelope.hpp>

namespace diffpy {
namespace srreal {

/// @class StepCutEnvelope
/// @brief empirical step-function PDF envelope.  The envelope is not
/// applied when the cutoff radius stepcut is smaller or equal zero.

class StepCutEnvelope : public PDFEnvelope
{
    public:

        // constructors
        StepCutEnvelope();
        virtual PDFEnvelope* create() const;
        virtual PDFEnvelope* copy() const;

        // methods

        virtual const std::string& type() const;
        virtual double operator()(const double& r) const;
        void setStepCut(double sc);
        const double& getStepCut() const;

    private:

        // data
        double mstepcut;

};  // class StepCutEnvelope

}   // namespace srreal
}   // namespace diffpy

#endif  // STEPCUTENVELOPE_HPP_INCLUDED
