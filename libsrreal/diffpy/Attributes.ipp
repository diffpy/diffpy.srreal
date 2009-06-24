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
        DoubleAttribute(T* obj, Getter g, Setter s)
        {
            mobject = obj;
            mgetter = g;
            msetter = s;
        }

        double getValue() const
        {
            double rv = (mobject->*mgetter)();
            return rv;
        }

        void setValue(double value)
        {
            if (!msetter)
            {
                const char* emsg =
                    "Cannot change value of read-only DoubleAttribute.";
                throw std::logic_error(emsg);
            }
            (mobject->*msetter)(value);
        }

    private:
        // data
        T* mobject;
        Getter mgetter;
        Setter msetter;

};

}   // namespace attributes

//////////////////////////////////////////////////////////////////////////////
// class Attributes
//////////////////////////////////////////////////////////////////////////////

// Template Public Methods ---------------------------------------------------

template <class T, class Getter, class Setter>
void Attributes::registerDoubleAttribute(
        const std::string& name, T* obj, Getter g, Setter s)
{
    using diffpy::attributes::DoubleAttribute;
    mdoubleattrs[name].reset(
        new DoubleAttribute<T, Getter, Setter>(obj, g, s)
        );
}

}   // namespace diffpy

// vim:ft=cpp:

#endif  // ATTRIBUTES_IPP_INCLUDED
