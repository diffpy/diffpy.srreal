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
*
* $Id$
*
*****************************************************************************/

#include <diffpy/srreal/PDFBaseline.hpp>
#include <diffpy/ClassRegistry.hpp>

using namespace std;
using diffpy::ClassRegistry;

namespace diffpy {
namespace srreal {

// Factory Functions ---------------------------------------------------------

PDFBaseline* createPDFBaseline(const string& tp)
{
    return ClassRegistry<PDFBaseline>::create(tp);
}


bool registerPDFBaseline(const PDFBaseline& ref)
{
    return ClassRegistry<PDFBaseline>::add(ref);
}


bool aliasPDFBaseline(const string& tp, const string& al)
{
    return ClassRegistry<PDFBaseline>::alias(tp, al);
}

set<string> getPDFBaselineTypes()
{
    return ClassRegistry<PDFBaseline>::getTypes();
}

}   // namespace srreal
}   // namespace diffpy

// End of file
