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
* class Attributes - interface for calling setter and getter methods using
*   their string names.
*
* $Id$
*
*****************************************************************************/

#include <sstream>

#include <diffpy/Attributes.hpp>

using namespace std;
namespace diffpy {

//////////////////////////////////////////////////////////////////////////////
// class DoubleAttribute
//////////////////////////////////////////////////////////////////////////////

// Public Methods ------------------------------------------------------------

double Attributes::getDoubleAttr(const string& name) const
{
    DoubleAttributeStorage::const_iterator ati = mdoubleattrs.find(name);
    if (ati == mdoubleattrs.end())  this->raiseInvalidAttribute(name);
    double rv = ati->second->getValue();
    return rv;
}


void Attributes::setDoubleAttr(const string& name, double value)
{
    DoubleAttributeStorage::iterator ati = mdoubleattrs.find(name);
    if (ati == mdoubleattrs.end())  this->raiseInvalidAttribute(name);
    ati->second->setValue(value);
}


bool Attributes::hasDoubleAttr(const string& name) const
{
    return mdoubleattrs.count(name);
}


set<string> Attributes::namesOfDoubleAttributes() const
{
    set<string> rv;
    DoubleAttributeStorage::const_iterator ati;
    for (ati = mdoubleattrs.begin(); ati != mdoubleattrs.end(); ++ati)
    {
        rv.insert(ati->first);
    }
    return rv;
}

// Protected Methods ---------------------------------------------------------

void Attributes::raiseInvalidAttribute(const std::string& name) const
{
    ostringstream emsg;
    emsg << "Invalid attribute name '" << name << "'.";
    throw invalid_argument(emsg.str());
}

}   // namespace diffpy

// End of file
