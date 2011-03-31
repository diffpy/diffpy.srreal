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

namespace srrealmodule {
namespace nswrap_PDFBaseline {

using namespace boost;
using namespace boost::python;
using namespace diffpy::srreal;

// docstrings ----------------------------------------------------------------

const char* doc_PDFBaseline = "\
FIXME\n\
";

const char* doc_PDFBaseline_create = "\
FIXME\n\
";

const char* doc_PDFBaseline_clone = "\
FIXME\n\
";

const char* doc_PDFBaseline_type = "\
FIXME\n\
";

const char* doc_PDFBaseline___call__ = "\
FIXME\n\
";

const char* doc_PDFBaseline__registerThisType = "\
FIXME\n\
";

const char* doc_PDFBaseline_createByType = "\
FIXME\n\
";

const char* doc_PDFBaseline_getRegisteredTypes = "\
Set of string identifiers for registered PDFBaseline classes.\n\
These are allowed arguments for the createByType static method.\n\
";

// wrappers ------------------------------------------------------------------

DECLARE_PYSET_FUNCTION_WRAPPER(PDFBaseline::getRegisteredTypes,
        getPDFBaselineTypes_asset)

// Helper class allows overload of the PDFBaseline methods from Python.

class PDFBaselineWrap :
    public PDFBaseline,
    public wrapper_srreal<PDFBaseline>
{
    public:

        // HasClassRegistry methods

        PDFBaselinePtr create() const
        {
            return this->get_pure_virtual_override("create")();
        }

        PDFBaselinePtr clone() const
        {
            return this->get_pure_virtual_override("clone")();
        }

        const std::string& type() const
        {
            python::object tp = this->get_pure_virtual_override("type")();
            mtype = python::extract<std::string>(tp);
            return mtype;
        }

        // own methods

        double operator()(const double& x) const
        {
            return this->get_pure_virtual_override("__call__")(x);
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
        noncopyable>("PDFBaseline", doc_PDFBaseline)
        .def("create", &PDFBaseline::create,
                doc_PDFBaseline_create)
        .def("clone", &PDFBaseline::clone,
                doc_PDFBaseline_clone)
        .def("type", &PDFBaseline::type,
                return_value_policy<copy_const_reference>(),
                doc_PDFBaseline_type)
        .def("__call__", &PDFBaseline::operator(),
                doc_PDFBaseline___call__)
        .def("_registerThisType", &PDFBaseline::registerThisType,
                doc_PDFBaseline__registerThisType)
        .def("createByType", &PDFBaseline::createByType,
                doc_PDFBaseline_createByType)
        .staticmethod("createByType")
        .def("getRegisteredTypes", getPDFBaselineTypes_asset,
                doc_PDFBaseline_getRegisteredTypes)
        .staticmethod("getRegisteredTypes")
        ;

    register_ptr_to_python<PDFBaselinePtr>();
}

}   // namespace srrealmodule

// End of file
