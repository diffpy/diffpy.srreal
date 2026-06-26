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
* Bindings to the PDFEnvelope class.  The business methods can be overloaded
* from Python to create custom PDF envelope functions.
*
*****************************************************************************/

#include <nanobind/nanobind.h>
#include <nanobind/trampoline.h>

#include <diffpy/srreal/PDFEnvelope.hpp>
#include <diffpy/srreal/QResolutionEnvelope.hpp>
#include <diffpy/srreal/ScaleEnvelope.hpp>
#include <diffpy/srreal/SphericalShapeEnvelope.hpp>
#include <diffpy/srreal/StepCutEnvelope.hpp>

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
namespace nswrap_PDFEnvelope {

using namespace diffpy::srreal;

// docstrings ----------------------------------------------------------------

const char* doc_PDFEnvelope = "\
Base class and registry for functions that return PDF scaling envelopes.\n\
";

const char* doc_PDFEnvelope___call__ = "\
Calculate PDF envelope at the specified r.\n\
\n\
r    -- atom distance in Angstroms where the scale envelope is evaluated.\n\
        Float or NumPy array.\n\
\n\
Return float or NumPy array.\n\
";

const char* doc_QResolutionEnvelope = "\
Gaussian dampening envelope function due to limited Q-resolution.\n\
\n\
Returns   exp(-qdamp * x**2).\n\
Returns   1  when qdamp is zero or negative.\n\
";

const char* doc_ScaleEnvelope = "\
Uniform scaling envelope function.\n\
\n\
Returns   scale.\n\
";

const char* doc_SphericalShapeEnvelope = "\
Dampening PDF envelope due to finite spherical particle shape.\n\
\n\
Returns   (1 - 1.5*tau + 0.5*tau**3), where tau = x/spdiameter.\n\
Returns   1 when spdiameter <= 0.\n\
";

const char* doc_StepCutEnvelope = "\n\
Step function damping envelope.\n\
\n\
Returns   1 when x <= stepcut or if stepcut <= 0.\n\
Returns   0 when x > stepcut.\n\
";

// wrappers ------------------------------------------------------------------

// Helper class allows overload of the PDFEnvelope methods from Python.

class PDFEnvelopeWrap :
    public PDFEnvelope,
    public PythonTrampolineTag
{
    public:

        NB_TRAMPOLINE(PDFEnvelope, 4);

        // HasClassRegistry methods

        PDFEnvelopePtr create() const override
        {
            nb::gil_scoped_acquire gil;
            nb::detail::ticket ticket(nb_trampoline, "create", true);

            if (!ticket.key.is_valid()) 
            {
                throw nb::type_error(
                    "pure virtual method PDFEnvelope.create() called"
                );
            }

            nb::object rv = nb_trampoline.base().attr(ticket.key)();
            return mconfigurator.fetch(rv);
        }

        PDFEnvelopePtr clone() const override
        {
            NB_OVERRIDE_PURE(clone);
        }

        const std::string& type() const override
        {
            nb::gil_scoped_acquire gil;
            nb::detail::ticket ticket(nb_trampoline, "type", true);

            if (!ticket.key.is_valid()) 
            {
                throw nb::type_error(
                    "pure virtual method PDFEnvelope.type() called"
                );
            }

            nb::object tp = nb_trampoline.base().attr(ticket.key)();
            mtype = nb::cast<std::string>(tp);
            return mtype;
        }

        // own methods

        double operator()(const double& x) const override
        {
            nb::gil_scoped_acquire gil;
            nb::detail::ticket ticket(nb_trampoline, "__call__", true);

            if (!ticket.key.is_valid()) 
            {
                throw nb::type_error(
                    "pure virtual method PDFEnvelope.__call__() called"
                );
            }

            nb::object rv = nb_trampoline.base().attr(ticket.key)(x);
            return nb::cast<double>(rv);
        }

    protected:

        // HasClassRegistry method

        void setupRegisteredObject(PDFEnvelopePtr p) const override
        {
            mconfigurator.setup(p);
        }

    private:

        mutable std::string mtype;
        wrapper_registry_configurator<PDFEnvelope> mconfigurator;

        // serialization
        friend class boost::serialization::access;
        template<class Archive>
            void serialize(Archive& ar, const unsigned int version)
        {
            using boost::serialization::base_object;
            ar & base_object<PDFEnvelope>(*this);
        }

};  // class PDFEnvelopeWrap


nb::object callnparray(const PDFEnvelope* obj, nb::object& x)
{
    NumPyArray_DoublePtr xx = extractNumPyDoubleArray(x);
    NumPyArray_DoublePtr yy = createNumPyDoubleArrayLike(xx.first);
    double* src = xx.second;
    double* last = xx.second + PyArray_Size(xx.first.ptr());
    double* dst = yy.second;
    for (; src != last; ++src, ++dst)  *dst = (*obj)(*src);
    return yy.first;
}

}   // namespace nswrap_PDFEnvelope

// Wrapper definition --------------------------------------------------------

void wrap_PDFEnvelope(nb::module_ &m)
{
    using namespace nswrap_PDFEnvelope;
    using diffpy::Attributes;

    nb::class_<PDFEnvelope, Attributes, PDFEnvelopeWrap>
        pdfenvelope(m, "PDFEnvelope", nb::dynamic_attr(), doc_PDFEnvelope);
    wrap_registry_methods(pdfenvelope)
        .def(nb::init<>())
        .def("__call__", callnparray,
                nb::arg("r_array"))
        .def("__call__", &PDFEnvelope::operator(),
                nb::arg("r"), doc_PDFEnvelope___call__)
        ;
        SerializationPickleSuite<
            PDFEnvelope,
            DICT_PICKLE,
            PDFEnvelopeWrap>::bind(pdfenvelope);

    nb::class_<QResolutionEnvelope, PDFEnvelope> qresenvelope(m,
            "QResolutionEnvelope", doc_QResolutionEnvelope);
    qresenvelope
        .def(nb::init<>())
        ;
        SerializationPickleSuite<QResolutionEnvelope, DICT_GUARD>::bind(qresenvelope);
    nb::class_<ScaleEnvelope, PDFEnvelope> scaleenvelope(m,
            "ScaleEnvelope", doc_ScaleEnvelope);
    scaleenvelope
        .def(nb::init<>())
        ;
        SerializationPickleSuite<ScaleEnvelope, DICT_GUARD>::bind(scaleenvelope);
    nb::class_<SphericalShapeEnvelope, PDFEnvelope> sphshapeenvelope(m,
            "SphericalShapeEnvelope", doc_SphericalShapeEnvelope);
    sphshapeenvelope
        .def(nb::init<>())
        ;
        SerializationPickleSuite<SphericalShapeEnvelope, DICT_GUARD>::bind(sphshapeenvelope);
    nb::class_<StepCutEnvelope, PDFEnvelope> stepcutenvelope(m,
            "StepCutEnvelope", doc_StepCutEnvelope);
    stepcutenvelope
        .def(nb::init<>())
        ;
        SerializationPickleSuite<StepCutEnvelope, DICT_GUARD>::bind(stepcutenvelope);

}

}   // namespace srrealmodule

// Serialization -------------------------------------------------------------

BOOST_CLASS_EXPORT(srrealmodule::nswrap_PDFEnvelope::PDFEnvelopeWrap)

// End of file
