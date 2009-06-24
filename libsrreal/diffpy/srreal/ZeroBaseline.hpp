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

#ifndef ZEROBASELINE_HPP_INCLUDED
#define ZEROBASELINE_HPP_INCLUDED

#include <diffpy/srreal/PDFBaseline.hpp>

namespace diffpy {
namespace srreal {

/// @class ZeroBaseline
/// @brief trivial zero baseline

class ZeroBaseline : public PDFBaseline
{
    public:

        // constructors
        PDFBaseline* create() const;
        PDFBaseline* copy() const;

        // methods
        const std::string& type() const;
        double operator()(const double& r) const;

};  // class ZeroBaseline

}   // namespace srreal
}   // namespace diffpy

#endif  // ZEROBASELINE_HPP_INCLUDED
