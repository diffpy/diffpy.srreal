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
* class PDFEnvelope -- abstract base class for PDF envelope functions
*
* $Id$
*
*****************************************************************************/

#include <diffpy/srreal/PDFEnvelope.hpp>
#include <diffpy/ClassRegistry.hpp>

using namespace std;
using diffpy::ClassRegistry;

namespace diffpy {
namespace srreal {

// Factory Functions ---------------------------------------------------------

PDFEnvelope* createPDFEnvelope(const string& tp)
{
    return ClassRegistry<PDFEnvelope>::create(tp);
}


bool registerPDFEnvelope(const PDFEnvelope& ref)
{
    return ClassRegistry<PDFEnvelope>::add(ref);
}


bool aliasPDFEnvelope(const string& tp, const string& al)
{
    return ClassRegistry<PDFEnvelope>::alias(tp, al);
}

set<string> getPDFEnvelopeTypes()
{
    return ClassRegistry<PDFEnvelope>::getTypes();
}

}   // namespace srreal
}   // namespace diffpy

// End of file
