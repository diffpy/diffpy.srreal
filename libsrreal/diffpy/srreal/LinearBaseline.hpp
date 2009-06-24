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

#ifndef LINEARBASELINE_HPP_INCLUDED
#define LINEARBASELINE_HPP_INCLUDED

#include <diffpy/srreal/PDFBaseline.hpp>

namespace diffpy {
namespace srreal {

/// @class LinearBaseline
/// @brief linear PDF baseline

class LinearBaseline : public PDFBaseline
{
    public:

        // constructors
        LinearBaseline();
        PDFBaseline* create() const;
        PDFBaseline* copy() const;

        // methods
        const std::string& type() const;
        double operator()(const double& r) const;
        void setSlope(double sc);
        const double& getSlope() const;

    private:

        // data
        double mslope;

};  // class LinearBaseline

}   // namespace srreal
}   // namespace diffpy

#endif  // LINEARBASELINE_HPP_INCLUDED
