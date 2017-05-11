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

#include "srreal_numpy_symbol.hpp"
// numpy/arrayobject.h needs to be included after srreal_numpy_symbol.hpp,
// which defines PY_ARRAY_UNIQUE_SYMBOL.  NO_IMPORT_ARRAY indicates
// import_array will be called in the extension module initializer.
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>

#include "srreal_converters.hpp"
#include "srreal_pickling.hpp"
#include "srreal_registry.hpp"

namespace srrealmodule {
namespace nswrap_PDFBaseline {

using namespace boost;
using namespace boost::python;
using namespace diffpy::srreal;

// docstrings ----------------------------------------------------------------

const char* doc_PDFBaseline = "\
Base class and registry for functions that calculate PDF baseline.\n\
";

const char* doc_PDFBaseline___call__ = "\
Calculate PDF baseline at the specified r.\n\
\n\
r    -- atom distance in Angstroms where the baseline is calculated.\n\
        Float or NumPy array.\n\
\n\
Return float or NumPy array.\n\
";

const char* doc_ZeroBaseline = "\
Trivial baseline function that is always zero, no baseline.\n\
";

const char* doc_LinearBaseline = "\
PDF baseline function equal to (slope * r).\n\
";

// wrappers ------------------------------------------------------------------

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


object callnparray(const PDFBaseline* obj, object& x)
{
    NumPyArray_DoublePtr xx = extractNumPyDoubleArray(x);
    NumPyArray_DoublePtr yy = createNumPyDoubleArrayLike(xx.first);
    double* src = xx.second;
    double* last = xx.second + PyArray_Size(xx.first.ptr());
    double* dst = yy.second;
    for (; src != last; ++src, ++dst)  *dst = (*obj)(*src);
    return yy.first;
}


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

    class_<PDFBaselineWrap, bases<Attributes>, noncopyable>
        pdfbaseline("PDFBaseline", doc_PDFBaseline);
    wrap_registry_methods(pdfbaseline)
        .def("__call__", callnparray,
                bp::arg("r_array"))
        .def("__call__", &PDFBaseline::operator(),
                bp::arg("r"), doc_PDFBaseline___call__)
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
