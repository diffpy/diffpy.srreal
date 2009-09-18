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
namespace attributes {

//////////////////////////////////////////////////////////////////////////////
// class DoubleAttribute
//////////////////////////////////////////////////////////////////////////////

// Public Methods ------------------------------------------------------------

double Attributes::getDoubleAttr(const string& name) const
{
    this->checkAttributeName(name);
    GetDoubleAttrVisitor vg(name);
    this->accept(vg);
    return vg.getValue();
}


void Attributes::setDoubleAttr(const string& name, double value)
{
    this->checkAttributeName(name);
    SetDoubleAttrVisitor vs(name, value);
    this->accept(vs);
}


bool Attributes::hasDoubleAttr(const string& name) const
{
    CountDoubleAttrVisitor vc(name);
    this->accept(vc);
    return vc.count();
}


set<string> Attributes::namesOfDoubleAttributes() const
{
    NamesOfDoubleAttributesVisitor vn;
    this->accept(vn);
    set<string> rv = vn.names();
    return rv;
}

// Private Methods -----------------------------------------------------------

void Attributes::checkAttributeName(const string& name) const
{
    CountDoubleAttrVisitor v(name);
    this->accept(v);
    if (v.count() == 0)
    {
        ostringstream emsg;
        emsg << "Invalid attribute name '" << name << "'.";
        throw invalid_argument(emsg.str());
    }
    else if (v.count() > 1)
    {
        ostringstream emsg;
        emsg << "Duplicate attribute '" << name << "'.";
        throw logic_error(emsg.str());
    }
}

// Visitor Classes -----------------------------------------------------------

// CountDoubleAttrVisitor

Attributes::CountDoubleAttrVisitor::
CountDoubleAttrVisitor(const string& name) :
    mname(name),
    mcount(0)
{ }


void Attributes::CountDoubleAttrVisitor::
visit(const Attributes& a)
{
    mcount += a.mdoubleattrs.count(mname);
}


int Attributes::CountDoubleAttrVisitor::
count() const
{
    return mcount;
}

// GetDoubleAttrVisitor

Attributes::GetDoubleAttrVisitor::
GetDoubleAttrVisitor(const string& name) :
    mname(name)
{ }


void Attributes::GetDoubleAttrVisitor::
visit(const Attributes& a)
{
    DoubleAttributeStorage::const_iterator ai;
    ai = a.mdoubleattrs.find(mname);
    if (ai != a.mdoubleattrs.end())
    {
        mvalue = ai->second->getValue(&a);
    }
}


double Attributes::GetDoubleAttrVisitor::
getValue() const
{
    return mvalue;
}

// SetDoubleAttrVisitor

Attributes::SetDoubleAttrVisitor::
SetDoubleAttrVisitor(const string& name, double value) :
    mname(name),
    mvalue(value)
{ }


void Attributes::SetDoubleAttrVisitor::
visit(Attributes& a)
{
    DoubleAttributeStorage::iterator ai;
    ai = a.mdoubleattrs.find(mname);
    if (ai != a.mdoubleattrs.end())
    {
        ai->second->setValue(&a, mvalue);
    }
}

// NamesOfDoubleAttributesVisitor

void Attributes::NamesOfDoubleAttributesVisitor::
visit(const Attributes& a)
{
    DoubleAttributeStorage::const_iterator ati;
    for (ati = a.mdoubleattrs.begin(); ati != a.mdoubleattrs.end(); ++ati)
    {
        mnames.insert(ati->first);
    }
}


const set<string>& Attributes::NamesOfDoubleAttributesVisitor::
names() const
{
    return mnames;
}


}   // namespace attributes
}   // namespace diffpy

// End of file
