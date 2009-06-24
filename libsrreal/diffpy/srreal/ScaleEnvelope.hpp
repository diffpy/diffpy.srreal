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
* class ScaleEnvelope -- constant scaling factor
*
* $Id$
*
*****************************************************************************/

#ifndef SCALEENVELOPE_HPP_INCLUDED
#define SCALEENVELOPE_HPP_INCLUDED

#include <diffpy/srreal/PDFEnvelope.hpp>

namespace diffpy {
namespace srreal {

/// @class ScaleEnvelope
/// @brief constant PDF scaling factor

class ScaleEnvelope : public PDFEnvelope
{
    public:

        // constructors
        ScaleEnvelope();
        PDFEnvelope* create() const;
        PDFEnvelope* copy() const;

        // methods
        const std::string& type() const;
        double operator()(const double& r) const;
        void setScale(double sc);
        const double& getScale() const;

    private:

        // data
        double mscale;

};  // class ScaleEnvelope

}   // namespace srreal
}   // namespace diffpy

#endif  // SCALEENVELOPE_HPP_INCLUDED
