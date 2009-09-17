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
#include <stdexcept>

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


class BaseAttributesVisitor
{
    public:

        virtual ~BaseAttributesVisitor() { }

        virtual void visit(const Attributes& a)
        {
            const char* emsg =
                "Visitor must be implemented in a derived class.";
            throw std::logic_error(emsg);
        }

        virtual void visit(Attributes& a)
        {
            visit(static_cast<const Attributes&>(a));
        }
};

/// @class Attributes
/// @brief implementation of attribute access.  The client classes
/// should derive from Attributes and register their setter and
/// getter methods in their constructors.

class Attributes
{
    public:

        // class needs to be virtual to allow dynamic_cast in getValue
        virtual ~Attributes()  { }

        // methods
        double getDoubleAttr(const std::string& name) const;
        void setDoubleAttr(const std::string& name, double value);
        bool hasDoubleAttr(const std::string& name) const;
        std::set<std::string> namesOfDoubleAttributes() const;
        // visitors
        virtual void accept(BaseAttributesVisitor& v)  { v.visit(*this); }
        virtual void accept(BaseAttributesVisitor& v) const  { v.visit(*this); }

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
        void checkAttributeName(const std::string& name) const;

        // visitor classes

        class CountDoubleAttrVisitor : public BaseAttributesVisitor
        {
            public:

                CountDoubleAttrVisitor(const std::string& name);
                virtual void visit(const Attributes& a);
                int count() const;

            private:

                // data
                const std::string& mname;
                int mcount;
        };


        class GetDoubleAttrVisitor : public BaseAttributesVisitor
        {
            public:

                GetDoubleAttrVisitor(const std::string& name);
                virtual void visit(const Attributes& a);
                double getValue() const;

            private:

                // data
                const std::string& mname;
                double mvalue;
        };


        class SetDoubleAttrVisitor : public BaseAttributesVisitor
        {
            public:

                SetDoubleAttrVisitor(const std::string& name, double value);
                virtual void visit(Attributes& a);

            private:

                // data
                const std::string& mname;
                double mvalue;
        };


        class NamesOfDoubleAttributesVisitor : public BaseAttributesVisitor
        {
            public:

                virtual void visit(const Attributes& a);
                const std::set<std::string>& names() const;

            private:

                // data
                std::set<std::string> mnames;
        };

};  // class Attributes


}   // namespace attributes
}   // namespace diffpy

// Implementation ------------------------------------------------------------

#include <diffpy/Attributes.ipp>

// make selected classes visible in diffpy namespace
namespace diffpy {
    using attributes::Attributes;
    using attributes::BaseAttributesVisitor;
}

#endif  // ATTRIBUTES_HPP_INCLUDED
