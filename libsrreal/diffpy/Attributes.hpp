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

#ifndef ATTRIBUTES_HPP_INCLUDED
#define ATTRIBUTES_HPP_INCLUDED

#include <string>
#include <set>
#include <map>

#include <boost/shared_ptr.hpp>

namespace diffpy {
namespace attributes {

class Attributes;

/// @class BaseDoubleAttribute
/// @brief abstract base class for accessing a particular double attribute

class BaseDoubleAttribute
{
    public:
        virtual ~BaseDoubleAttribute() { }
        virtual double getValue(const Attributes* obj) const = 0;
        virtual void setValue(Attributes* obj, double value) = 0;
};

/// @class Attributes
/// @brief implementation of attribute access.  The client classes
/// should derive from Attributes and register their setter and
/// getter methods in their constructors.

class Attributes
{
    public:

        // class needs to be virtual to allow dynamic_cast in getValue
        virtual ~Attributes() { }

        // methods
        double getDoubleAttr(const std::string& name) const;
        void setDoubleAttr(const std::string& name, double value);
        bool hasDoubleAttr(const std::string& name) const;
        std::set<std::string> namesOfDoubleAttributes() const;

    protected:

        template <class T, class Getter>
            void registerDoubleAttribute(const std::string& name, T* obj, Getter);
        template <class T, class Getter, class Setter>
            void registerDoubleAttribute(const std::string& name, T* obj, Getter, Setter);

    private:

        // types
        typedef std::map<std::string,
                boost::shared_ptr<attributes::BaseDoubleAttribute> >
                    DoubleAttributeStorage;
        // data
        DoubleAttributeStorage mdoubleattrs;

        // methods
        void raiseInvalidAttribute(const std::string& name) const;

};

}   // namespace attributes
}   // namespace diffpy

// Implementation ------------------------------------------------------------

#include <diffpy/Attributes.ipp>

// make the Attributes class visible in diffpy namespace
namespace diffpy { using attributes::Attributes; }

#endif  // ATTRIBUTES_HPP_INCLUDED
