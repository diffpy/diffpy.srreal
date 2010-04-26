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
* Bindings to the PDFBaseline class.  The business methods can be overloaded
* from Python to create custom PDF baseline functions.
*
* $Id$
*
*****************************************************************************/

#include <boost/python.hpp>

#include <diffpy/srreal/PDFBaseline.hpp>

#include "srreal_converters.hpp"
#include "srreal_docstrings.hpp"

namespace srrealmodule {
namespace nswrap_PDFBaseline {

using namespace boost;
using namespace boost::python;
using namespace diffpy::srreal;

DECLARE_PYSET_FUNCTION_WRAPPER(PDFBaseline::getRegisteredTypes,
        getPDFBaselineTypes_asset)

// Helper class allows overload of the PDFBaseline methods from Python.

class PDFBaselineWrap :
    public PDFBaseline,
    public wrapper<PDFBaseline>
{
    public:

        // constructors

        PDFBaselinePtr create() const
        {
            return this->get_override("create")();
        }

        PDFBaselinePtr clone() const
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

};  // class PDFBaselineWrap

}   // namespace nswrap_PDFBaseline

// Wrapper definition --------------------------------------------------------

void wrap_PDFBaseline()
{
    using namespace nswrap_PDFBaseline;
    using diffpy::Attributes;

    class_<PDFBaselineWrap, bases<Attributes>,
        noncopyable>("PDFBaseline_ext")
        .def("create", pure_virtual(&PDFBaseline::create))
        .def("clone", pure_virtual(&PDFBaseline::clone))
        .def("type", pure_virtual(&PDFBaseline::type),
                return_value_policy<copy_const_reference>())
        .def("__call__", pure_virtual(&PDFBaseline::operator()))
        .def("_registerThisType", &PDFBaseline::registerThisType)
        .def("createByType", &PDFBaseline::createByType)
        .staticmethod("createByType")
        .def("getRegisteredTypes", getPDFBaselineTypes_asset,
                doc_PDFBaseline_getRegisteredTypes)
        .staticmethod("getRegisteredTypes")
        ;

    register_ptr_to_python<PDFBaselinePtr>();
}

}   // namespace srrealmodule

// End of file
