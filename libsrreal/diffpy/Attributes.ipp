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

#ifndef ATTRIBUTES_IPP_INCLUDED
#define ATTRIBUTES_IPP_INCLUDED

#include <stdexcept>
#include <cassert>

namespace diffpy {
namespace attributes {

//////////////////////////////////////////////////////////////////////////////
// class DoubleAttribute
//////////////////////////////////////////////////////////////////////////////

template <class T, class Getter, class Setter>
class DoubleAttribute : public BaseDoubleAttribute
{
    public:

        // constructor
        DoubleAttribute(Getter g, Setter s)
        {
            mgetter = g;
            msetter = s;
        }

        double getValue(const Attributes* obj) const
        {
            const T* tobj = dynamic_cast<const T*>(obj);
            assert(tobj);
            double rv = (tobj->*mgetter)();
            return rv;
        }

        void setValue(Attributes* obj, double value)
        {
            if (!msetter)
            {
                const char* emsg =
                    "Cannot change value of read-only DoubleAttribute.";
                // FIXME: replace with custom attribute exception
                throw std::logic_error(emsg);
            }
            T* tobj = dynamic_cast<T*>(obj);
            assert(tobj);
            (tobj->*msetter)(value);
        }

    private:
        // data
        Getter mgetter;
        Setter msetter;

};

//////////////////////////////////////////////////////////////////////////////
// class Attributes
//////////////////////////////////////////////////////////////////////////////

// Template Protected Methods ------------------------------------------------

template <class T, class Getter>
void Attributes::registerDoubleAttribute(
        const std::string& name, T* obj, Getter g)
{
    typedef  void (T::*Setter)(double x);
    mdoubleattrs[name].reset(new DoubleAttribute<T, Getter, Setter>(g, NULL));
}


template <class T, class Getter, class Setter>
void Attributes::registerDoubleAttribute(
        const std::string& name, T* obj, Getter g, Setter s)
{
    mdoubleattrs[name].reset(
        new DoubleAttribute<T, Getter, Setter>(g, s)
        );
}

}   // namespace attributes
}   // namespace diffpy

// vim:ft=cpp:

#endif  // ATTRIBUTES_IPP_INCLUDED
