/*****************************************************************************
*
* diffpy.srreal     by DANSE Diffraction group
*                   Simon J. L. Billinge
*                   (c) 2010 The Trustees of Columbia University
*                   in the City of New York.  All rights reserved.
*
* File coded by:    Pavol Juhas
*
* See AUTHORS.txt for a list of people who contributed.
* See LICENSE_DANSE.txt for license information.
*
******************************************************************************
*
* Bindings to the PDFBaseline class.  The business methods can be overloaded
* from Python to create custom PDF baseline functions.
*
*****************************************************************************/

#include <boost/python/class.hpp>
#include <boost/python/def.hpp>
#include <boost/python/copy_const_reference.hpp>
#include <boost/python/register_ptr_to_python.hpp>

#include <diffpy/srreal/PDFBaseline.hpp>
#include <diffpy/srreal/ZeroBaseline.hpp>
#include <diffpy/srreal/LinearBaseline.hpp>

#include "srreal_converters.hpp"
#include "srreal_pickling.hpp"

namespace srrealmodule {
namespace nswrap_PDFBaseline {

using namespace boost;
using namespace boost::python;
using namespace diffpy::srreal;

// docstrings ----------------------------------------------------------------

const char* doc_PDFBaseline = "\
Base class and registry for functions that calculate PDF baseline.\n\
";

const char* doc_PDFBaseline_create = "\
Return a new instance of the same type as self.\n\
\n\
This method must be overloaded in a derived class.\n\
";

const char* doc_PDFBaseline_clone = "\
Return a new instance that is a copy of self.\n\
\n\
This method must be overloaded in a derived class.\n\
";

const char* doc_PDFBaseline_type = "\
Return a unique string type that identifies a PDFBaseline-derived class.\n\
The string type is used for class registration and in the createByType\n\
function.\n\
\n\
This method must be overloaded in a derived class.\n\
";

const char* doc_PDFBaseline___call__ = "\
Calculate PDF baseline at the specified r.\n\
\n\
r    -- atom distance in Angstroms where the baseline is calculated.\n\
\n\
Return float.\n\
";

const char* doc_PDFBaseline__registerThisType = "\
Add this class to the global registry of PDFBaseline types.\n\
\n\
This method must be called once after definition of the derived\n\
class to support pickling and the createByType factory.\n\
";

const char* doc_PDFBaseline_createByType = "\
Return a new PDFBaseline instance of the specified string type.\n\
\n\
tp   -- string type identifying a registered PDFBaseline class\n\
        See getRegisteredTypes for the allowed values.\n\
\n\
Return a new instance of the PDFBaseline-derived class.\n\
";

const char* doc_PDFBaseline_getRegisteredTypes = "\
Return a set of string types of the registered PDFBaseline classes.\n\
These are the allowed arguments for the createByType factory.\n\
";

const char* doc_ZeroBaseline = "\
Trivial baseline function that is always zero, no baseline.\n\
";

const char* doc_LinearBaseline = "\
PDF baseline function equal to (slope * r).\n\
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
            object rv = this->get_pure_virtual_override("create")();
            return mconfigurator.fetch(rv);
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

    protected:

        // HasClassRegistry method

        void setupRegisteredObject(PDFBaselinePtr p) const
        {
            mconfigurator.setup(p);
        }

    private:

        mutable std::string mtype;
        wrapper_registry_configurator<PDFBaseline> mconfigurator;

};  // class PDFBaselineWrap


std::string baseline_tostring(PDFBaselinePtr obj)
{
    return diffpy::serialization_tostring(obj);
}


PDFBaselinePtr baseline_fromstring(std::string content)
{
    PDFBaselinePtr rv;
    diffpy::serialization_fromstring(rv, content);
    return rv;
}

}   // namespace nswrap_PDFBaseline

// Wrapper definition --------------------------------------------------------

void wrap_PDFBaseline()
{
    using namespace nswrap_PDFBaseline;
    using diffpy::Attributes;
    namespace bp = boost::python;

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
                bp::arg("r"), doc_PDFBaseline___call__)
        .def("_registerThisType", &PDFBaseline::registerThisType,
                doc_PDFBaseline__registerThisType)
        .def("createByType", &PDFBaseline::createByType,
                bp::arg("tp"), doc_PDFBaseline_createByType)
        .staticmethod("createByType")
        .def("getRegisteredTypes", getPDFBaselineTypes_asset,
                doc_PDFBaseline_getRegisteredTypes)
        .staticmethod("getRegisteredTypes")
        .enable_pickling()
        ;

    register_ptr_to_python<PDFBaselinePtr>();

    class_<ZeroBaseline, bases<PDFBaseline> >(
            "ZeroBaseline", doc_ZeroBaseline);
    class_<LinearBaseline, bases<PDFBaseline> >(
            "LinearBaseline", doc_ZeroBaseline);

    // pickling support functions
    def("_PDFBaseline_tostring", baseline_tostring);
    def("_PDFBaseline_fromstring", baseline_fromstring);

}

}   // namespace srrealmodule

// End of file
