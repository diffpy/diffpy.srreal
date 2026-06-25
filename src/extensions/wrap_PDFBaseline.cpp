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

#include <nanobind/nanobind.h>
#include <nanobind/trampoline.h>

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

namespace nb = nanobind;

namespace srrealmodule {
namespace nswrap_PDFBaseline {

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
    public PDFBaseline
{
    public:

        NB_TRAMPOLINE(PDFBaseline, 4);

        // HasClassRegistry methods

        PDFBaselinePtr create() const
        {
            nb::gil_scoped_acquire gil;
            nb::detail::ticket ticket(nb_trampoline, "create", true);

            if (!ticket.key.is_valid()) 
            {
                throw nb::type_error(
                    "pure virtual method PDFBaseline.create() called"
                );
            }

            nb::object rv = nb_trampoline.base().attr(ticket.key)();
            return mconfigurator.fetch(rv);
        }

        PDFBaselinePtr clone() const
        {
            NB_OVERRIDE_PURE(clone);
        }

        const std::string& type() const
        {
            nb::gil_scoped_acquire gil;
            nb::detail::ticket ticket(nb_trampoline, "type", true);

            if (!ticket.key.is_valid()) 
            {
                throw nb::type_error(
                    "pure virtual method PDFBaseline.type() called"
                );
            }

            nb::object tp = nb_trampoline.base().attr(ticket.key)();
            mtype = nb::cast<std::string>(tp);
            return mtype;
        }

        // own methods

        double operator()(const double& x) const
        {
            NB_OVERRIDE_PURE_NAME("__call__", operator(), x);
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

        // serialization
        friend class boost::serialization::access;
        template<class Archive>
            void serialize(Archive& ar, const unsigned int version)
        {
            using boost::serialization::base_object;
            ar & base_object<PDFBaseline>(*this);
        }

};  // class PDFBaselineWrap


nb::object callnparray(const PDFBaseline* obj, nb::object& x)
{
    NumPyArray_DoublePtr xx = extractNumPyDoubleArray(x);
    NumPyArray_DoublePtr yy = createNumPyDoubleArrayLike(xx.first);
    double* src = xx.second;
    double* last = xx.second + PyArray_Size(xx.first.ptr());
    double* dst = yy.second;
    for (; src != last; ++src, ++dst)  *dst = (*obj)(*src);
    return yy.first;
}

}   // namespace nswrap_PDFBaseline

// Wrapper definition --------------------------------------------------------

void wrap_PDFBaseline(nb::module_& m)
{
    using namespace nswrap_PDFBaseline;
    using diffpy::Attributes;

    nb::class_<PDFBaseline, Attributes, PDFBaselineWrap>
        pdfbaseline(m, "PDFBaseline", nb::dynamic_attr(), doc_PDFBaseline);
    wrap_registry_methods(pdfbaseline)
        .def(nb::init<>())
        .def("__call__", callnparray,
                nb::arg("r_array"))
        .def("__call__", &PDFBaseline::operator(),
                nb::arg("r"), doc_PDFBaseline___call__)
        ;
        SerializationPickleSuite<PDFBaseline, DICT_PICKLE>::bind(pdfbaseline);

    nb::class_<ZeroBaseline, PDFBaseline>
        zerobaseline(m, "ZeroBaseline", doc_ZeroBaseline);
    zerobaseline
        .def(nb::init<>());
        SerializationPickleSuite<ZeroBaseline, DICT_IGNORE>::bind(zerobaseline);
    nb::class_<LinearBaseline, PDFBaseline>
        linearbaseline(m,"LinearBaseline", doc_LinearBaseline);
    linearbaseline
        .def(nb::init<>());
        SerializationPickleSuite<LinearBaseline, DICT_IGNORE>::bind(linearbaseline);

}

}   // namespace srrealmodule

// Serialization -------------------------------------------------------------

BOOST_CLASS_EXPORT(srrealmodule::nswrap_PDFBaseline::PDFBaselineWrap)

// End of file
