/*****************************************************************************
*
* diffpy.srreal     by DANSE Diffraction group
*                   Simon J. L. Billinge
*                   (c) 2010 Trustees of the Columbia University
*                   in the City of New York.  All rights reserved.
*
* File coded by:    Pavol Juhas
*
* See AUTHORS.txt for a list of people who contributed.
* See LICENSE.txt for license information.
*
******************************************************************************
*
* Bindings to the PDFEnvelope class.  The business methods can be overloaded
* from Python to create custom PDF envelope functions.
*
* $Id$
*
*****************************************************************************/

#include <boost/python.hpp>

#include <diffpy/srreal/PDFEnvelope.hpp>

#include "srreal_converters.hpp"
#include "srreal_docstrings.hpp"

namespace srrealmodule {
namespace nswrap_PDFEnvelope {

using namespace boost;
using namespace boost::python;
using namespace diffpy::srreal;

DECLARE_PYSET_FUNCTION_WRAPPER(PDFEnvelope::getRegisteredTypes,
        getPDFEnvelopeTypes_asset)

// Helper class allows overload of the PDFEnvelope methods from Python.

class PDFEnvelopeWrap :
    public PDFEnvelope,
    public wrapper<PDFEnvelope>
{
    public:

        // constructors

        PDFEnvelopePtr create() const
        {
            return this->get_override("create")();
        }

        PDFEnvelopePtr clone() const
        {
            return this->get_override("clone")();
        }

        // methods

        const std::string& type() const
        {
            python::object tp = this->get_override("type")();
            mtype = python::extract<std::string>(tp);
            return mtype;
        }

        double operator()(const double& x) const
        {
            return this->get_override("__call__")(x);
        }

    private:

        mutable std::string mtype;

};  // class PDFEnvelopeWrap

}   // namespace nswrap_PDFEnvelope

// Wrapper definition --------------------------------------------------------

void wrap_PDFEnvelope()
{
    using namespace nswrap_PDFEnvelope;
    using diffpy::Attributes;

    class_<PDFEnvelopeWrap, bases<Attributes>,
        noncopyable>("PDFEnvelope_ext")
        .def("create", pure_virtual(&PDFEnvelope::create))
        .def("clone", pure_virtual(&PDFEnvelope::clone))
        .def("type", pure_virtual(&PDFEnvelope::type),
                return_value_policy<copy_const_reference>())
        .def("__call__", pure_virtual(&PDFEnvelope::operator()))
        .def("_registerThisType", &PDFEnvelope::registerThisType)
        .def("createByType", &PDFEnvelope::createByType)
        .staticmethod("createByType")
        .def("getRegisteredTypes", getPDFEnvelopeTypes_asset,
                doc_PDFEnvelope_getRegisteredTypes)
        .staticmethod("getRegisteredTypes")
        ;

    register_ptr_to_python<PDFEnvelopePtr>();
}

}   // namespace srrealmodule

// End of file
