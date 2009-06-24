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
* class PDFBaseline -- abstract base class for PDF baseline functions
*     A concrete instance of PDFBaseline is a functor, that calculates
*     baseline value at a given pair distance r.  The baseline is added to
*     (R(r) * r) before multiplication by any envelope functions.
*
* $Id$
*
*****************************************************************************/

#ifndef PDFBASELINE_HPP_INCLUDED
#define PDFBASELINE_HPP_INCLUDED

#include <string>
#include <set>

#include <diffpy/Attributes.hpp>

namespace diffpy {
namespace srreal {

/// @class PDFBaseline
/// @brief abstract base class for PDF baseline function

class PDFBaseline : public diffpy::Attributes
{
    public:

        // constructors
        virtual PDFBaseline* create() const = 0;
        virtual PDFBaseline* copy() const = 0;
        virtual ~PDFBaseline()  { }

        // methods
        virtual const std::string& type() const = 0;
        virtual double operator()(const double& r) const = 0;
};

// Factory functions for concrete PDF envelopes ------------------------------

PDFBaseline* createPDFBaseline(const std::string& tp);
bool registerPDFBaseline(const PDFBaseline&);
bool aliasPDFBaseline(const std::string& tp, const std::string& al);
std::set<std::string> getPDFBaselineTypes();

}   // namespace srreal
}   // namespace diffpy

#endif  // PDFBASELINE_HPP_INCLUDED
