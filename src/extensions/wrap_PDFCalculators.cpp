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
* Bindings to DebyePDFCalculator and PDFCalculator classes.
*
*****************************************************************************/

#include <nanobind/nanobind.h>

#include <diffpy/srreal/DebyePDFCalculator.hpp>
#include <diffpy/srreal/PDFCalculator.hpp>

#include "srreal_converters.hpp"
#include "srreal_pickling.hpp"

namespace nb = nanobind;

namespace srrealmodule {
namespace nswrap_PDFCalculators {

using namespace diffpy::srreal;

// docstrings ----------------------------------------------------------------

const char* doc_PDFCommon_pdf = "\
An array of PDF values in the form of G = 4*pi*r(rho - rho0) in A**-2.\n\
";

const char* doc_PDFCommon_rdf = "\
An array of the last RDF values in A**-1.\n\
";

const char* doc_PDFCommon_rgrid = "\
An array of r-values in Angstrom.  This is a uniformly spaced array of\n\
rstep multiples that are greater or equal to rmin and smaller than rmax.\n\
";

const char* doc_PDFCommon_fq = "\
An array of F values in 1/A that can be Fourier transformed to G(r).\n\
";

const char* doc_PDFCommon_qgrid = "\
An array of Q-values in 1/A.  This is a uniformly spaced array of qstep\n\
values that start at 0/A and are smaller than qmax.\n\
";

const char* doc_PDFCommon_envelopes = "\
A tuple of PDFEnvelope instances used for calculating scaling envelope.\n\
This property can be assigned an iterable of PDFEnvelope objects.\n\
";

const char* doc_PDFCommon_usedenvelopetypes = "\
A tuple of string types of the used PDFEnvelope instances, read-only.\n\
";

const char* doc_PDFCommon_addEnvelope = "\
Add a PDFEnvelope object as another scaling function.\n\
This replaces any existing PDFEnvelope of the same string type.\n\
\n\
envlp    -- instance of PDFEnvelope that defines the scaling function or\n\
            a string type for a registered PDFEnvelope class.\n\
            Use PDFEnvelope.getRegisteredTypes for the available\n\
            string values.\n\
\n\
No return value.\n\
";

const char* doc_PDFCommon_popEnvelope = "\
Remove PDFEnvelope object from an internal list of scaling functions.\n\
\n\
envlp    -- instance of PDFEnvelope object or a string type of a registered\n\
            PDFEnvelope class to be removed.  No action if envlp is not\n\
            present in the calculator.  See the 'envelopes' attribute for a\n\
            tuple of active PDFEnvelope instances or the 'usedenvelopetypes'\n\
            attribute for the corresponding string types.\n\
\n\
No return value.\n\
";

const char* doc_PDFCommon_getEnvelope = "\
Retrieve an active PDFEnvelope object by its string type.\n\
\n\
tp   -- string type of a PDFEnvelope object that is present in\n\
        the calculator.  See the 'usedenvelopetypes' attribute\n\
        for the present string types.\n\
\n\
Return a PDFEnvelope instance.\n\
Raise ValueError it type tp is not present.\n\
";

const char* doc_PDFCommon_clearEnvelopes = "\
Remove all PDFEnvelope scaling functions from the calculator.\n\
";

const char* doc_DebyePDFCalculator = "\
Calculate PDF using the Debye scattering equation.\n\
";

const char* doc_DebyePDFCalculator_setOptimumQstep = "\
Use the optimum qstep value equal to the Nyquist step of pi/rmaxext,\n\
where rmaxext is rmax extended for termination ripples and peak tails.\n\
The qstep value depends on rmax when active.  This is disabled after\n\
explicit qstep assignment, which makes qstep independent of rmax.\n\
";

const char* doc_DebyePDFCalculator_isOptimumQstep = "\
Return True if qstep is set to an optimum, rmax-dependent value.\n\
Return False if qstep was overridden by the user.\n\
";

const char* doc_PDFCalculator = "\
Calculate PDF using the real-space summation of PeakProfile functions.\n\
";

const char* doc_PDFCalculator_peakprofile = "\
Instance of PeakProfile that calculates the real-space profile for\n\
a single atom-pair contribution.  This can be assigned either a\n\
PeakProfile-derived object or a string type of a registered PeakProfile\n\
class.  Use PeakProfile.getRegisteredTypes() for the allowed values.\n\
";

const char* doc_PDFCalculator_baseline = "\
Instance of PDFBaseline that calculates unscaled baseline at r.\n\
The baseline property can be assigned either a PDFBaseline-derived\n\
object or a string type of a registered PDFBaseline class.\n\
Use PDFBaseline.getRegisteredTypes() for the set of allowed values.\n\
";

const char* doc_fftftog = "\
Perform sine-fast Fourier transform from F(Q) to G(r).\n\
The length of the output array is padded to the next power of 2.\n\
\n\
f        -- array of the F values on a regular Q-space grid.\n\
qstep    -- spacing in the Q-space grid, this is used for proper\n\
            scaling of the output array.\n\
qmin     -- optional starting point of the Q-space grid.\n\
\n\
Return a tuple of (g, rstep).  These can be used with the complementary\n\
fftgtof function to recover the original signal f.\n\
";

const char* doc_fftgtof = "\
Perform sine-fast Fourier transform from G(r) to F(Q).\n\
The length of the output array is padded to the next power of 2.\n\
\n\
g        -- array of the G values on a regular r-space grid.\n\
rstep    -- spacing in the r-space grid, this is used for proper\n\
            scaling of the output array.\n\
rmin     -- optional starting point of the r-space grid.\n\
\n\
Return a tuple of (f, qstep).  These can be used with the complementary\n\
fftftog function to recover the original signal g.\n\
";

const char* doc_PeakWidthModelOwner_peakwidthmodel = "\
PeakWidthModel object used for calculating the FWHM of the PDF peaks.\n\
This attribute can be assigned either a PeakWidthModel-derived object\n\
or a string name of a registered PeakWidthModel class.\n\
Use PeakWidthModel.getRegisteredTypes() for a set of registered names.\n\
";

const char* doc_ScatteringFactorTableOwner_scatteringfactortable = "\
ScatteringFactorTable object used for a lookup of scattering factors.\n\
This can be also set with the setScatteringFactorTableByType method.\n\
";

const char* doc_ScatteringFactorTableOwner_setScatteringFactorTableByType = "\
Set internal ScatteringFactorTable according to specified string type.\n\
\n\
tp   -- string identifier of a registered ScatteringFactorTable type.\n\
    Use ScatteringFactorTable.getRegisteredTypes for the allowed values.\n\
\n\
Deprecated: This method is deprecated and will be removed in the 2.0.0 release.\n\
Use direct assignment to the `scatteringfactortable` property instead, for example:\n\
    obj.scatteringfactortable = SFTNeutron()\n\
No return value.\n\
";

const char* doc_ScatteringFactorTableOwner_getRadiationType = "\
Return string identifying the radiation type.\n\
'X' for x-rays, 'N' for neutrons.\n\
";

// wrappers ------------------------------------------------------------------

DECLARE_PYARRAY_METHOD_WRAPPER(getPDF, getPDF_asarray)
DECLARE_PYARRAY_METHOD_WRAPPER(getRDF, getRDF_asarray)
DECLARE_PYARRAY_METHOD_WRAPPER(getRgrid, getRgrid_asarray)
DECLARE_PYARRAY_METHOD_WRAPPER(getF, getF_asarray)
DECLARE_PYARRAY_METHOD_WRAPPER(getQgrid, getQgrid_asarray)
DECLARE_PYLIST_METHOD_WRAPPER(usedEnvelopeTypes, usedEnvelopeTypes_aslist)

// wrappers for the peakprofile property

PeakProfilePtr getpeakprofile(PDFCalculator& obj)
{
    return obj.getPeakProfile();
}

DECLARE_BYTYPE_SETTER_WRAPPER(setPeakProfile, setpeakprofile)

// wrappers for the baseline property

PDFBaselinePtr getbaseline(PDFCalculator& obj)
{
    return obj.getBaseline();
}

DECLARE_BYTYPE_SETTER_WRAPPER(setBaseline, setbaseline)

// wrappers for the envelopes property

PDFEnvelopePtr pyobjtoenvelope(nb::object evp)
{
    std::string tp;
    if (nb::try_cast<std::string>(evp, tp, false))
    {
        return PDFEnvelope::createByType(tp);
    }
    return nb::cast<PDFEnvelopePtr>(evp);
}


template <class T>
nb::tuple getenvelopes(T& obj)
{
    std::set<std::string> etps = obj.usedEnvelopeTypes();
    std::set<std::string>::const_iterator tpi;
    nb::list elst;
    for (tpi = etps.begin(); tpi != etps.end(); ++tpi)
    {
        elst.append(obj.getEnvelopeByType(*tpi));
    }
    return nb::tuple(elst);
}

template <class T>
void setenvelopes(T& obj, nb::object evps)
{
    // first check if all evps items can be converted to PDFEnvelopePtr
    std::list<PDFEnvelopePtr> elst;
    for (nb::handle evp : evps)  elst.push_back(pyobjtoenvelope(nb::borrow<nb::object>(evp)));
    // if that worked, overwrite the original envelopes
    obj.clearEnvelopes();
    std::list<PDFEnvelopePtr>::iterator eii = elst.begin();
    for (; eii != elst.end(); ++eii)  obj.addEnvelope(*eii);
}

// wrapper for the usedenvelopetypes

template <class T>
nb::tuple getusedenvelopetypes(T& obj)
{
    nb::tuple rv(usedEnvelopeTypes_aslist<T>(obj));
    return rv;
}

template <class T>
void addenvelope(T& obj, nb::object evp)
{
    PDFEnvelopePtr e = pyobjtoenvelope(evp);
    obj.addEnvelope(e);
}

template <class T>
void popenvelope(T& obj, nb::object evp)
{
    std::string tp;
    if (nb::try_cast<std::string>(evp, tp, false))
    {
        obj.popEnvelopeByType(tp);
    }
    else
    {
        PDFEnvelopePtr e = nb::cast<PDFEnvelopePtr>(evp);
        obj.popEnvelope(e);
    }
}

template <class T>
PDFEnvelopePtr getoneenvelope(T& obj, const std::string& tp)
{
    return obj.getEnvelopeByType(tp);
}


// wrappers for PeakWidthModelOwner behavior

template <class T>
PeakWidthModelPtr getpeakwidthmodel(T& obj)
{
    return obj.getPeakWidthModel();
}

DECLARE_BYTYPE_SETTER_WRAPPER(setPeakWidthModel, setpeakwidthmodel)

// wrappers for ScatteringFactorTableOwner behavior

template <class T>
ScatteringFactorTablePtr getscatteringfactortable(T& obj)
{
    return obj.getScatteringFactorTable();
}

DECLARE_BYTYPE_SETTER_WRAPPER(setScatteringFactorTable, setscatteringfactortable)

template <class T>
void setscatteringfactortablebytype(T& obj, const std::string& tp)
{
    try
    {
        nb::object warnings = nb::module_::import_("warnings");
        nb::object builtins = nb::module_::import_("builtins");
        nb::object DeprecationWarning = builtins.attr("DeprecationWarning");
        warnings.attr("warn")(
            std::string("setScatteringFactorTableByType is deprecated; "
                    "assign the 'scatteringfactortable' property directly, for example:\n"
                    "obj.scatteringfactortable = SFTNeutron()"),
            DeprecationWarning,
            2);
    }
    catch (...) { /* don't let warnings break the binding */ }
    obj.setScatteringFactorTableByType(tp);
}

template <class T>
std::string getradiationtype(T& obj)
{
    return obj.getRadiationType();
}


// wrap shared methods and attributes of PDFCalculators
// TODO: since nanobind doesn't allow multiple inheritance,
// we may need to introduce a header to better address this.

template <class W, class C>
C& wrap_PDFCommon(C& cls)
{
    cls
        // result vectors
        .def_prop_ro("pdf", getPDF_asarray<W>, doc_PDFCommon_pdf)
        .def_prop_ro("rdf", getRDF_asarray<W>, doc_PDFCommon_rdf)
        .def_prop_ro("rgrid", getRgrid_asarray<W>, doc_PDFCommon_rgrid)
        .def_prop_ro("fq", getF_asarray<W>, doc_PDFCommon_fq)
        .def_prop_ro("qgrid", getQgrid_asarray<W>, doc_PDFCommon_qgrid)
        // PDF envelopes
        .def_prop_rw("envelopes",
                getenvelopes<W>, setenvelopes<W>,
                doc_PDFCommon_envelopes)
        .def_prop_ro("usedenvelopetypes",
                getusedenvelopetypes<W>,
                doc_PDFCommon_usedenvelopetypes)
        .def("addEnvelope", addenvelope<W>,
                nb::arg("envlp"), doc_PDFCommon_addEnvelope)
        .def("popEnvelope", popenvelope<W>,
                nb::arg("envlp"), doc_PDFCommon_popEnvelope)
        .def("getEnvelope", getoneenvelope<W>,
                nb::arg("tp"), doc_PDFCommon_getEnvelope)
        .def("clearEnvelopes", &W::clearEnvelopes,
                doc_PDFCommon_clearEnvelopes)
        // PeakWidthModelOwner functions
        .def_prop_rw("peakwidthmodel",
                getpeakwidthmodel<W>,
                setpeakwidthmodel<W, PeakWidthModel>,
                doc_PeakWidthModelOwner_peakwidthmodel)
        // ScatteringFactorTableOwner behavior
        .def_prop_rw("scatteringfactortable",
                getscatteringfactortable<W>,
                setscatteringfactortable<W, ScatteringFactorTable>,
                doc_ScatteringFactorTableOwner_scatteringfactortable)
        .def("setScatteringFactorTableByType",
                setscatteringfactortablebytype<W>,
                nb::arg("tp"),
                doc_ScatteringFactorTableOwner_setScatteringFactorTableByType)
        .def("getRadiationType",
                getradiationtype<W>,
                doc_ScatteringFactorTableOwner_getRadiationType)
        ;
    return cls;
}

// local helper for converting python object to a quantity type

nb::tuple fftftog_array_step(nb::object f, double qstep, double qmin)
{
    QuantityType f0;
    const QuantityType& f1 = extractQuantityType(f, f0);
    QuantityType g = fftftog(f1, qstep, qmin);
    nb::object ga = convertToNumPyArray(g);
    double qmaxpad = g.size() * qstep;
    double rstep = (qmaxpad > 0) ? (M_PI / qmaxpad) : 0.0;
    return nb::make_tuple(ga, rstep);
}


nb::tuple fftgtof_array_step(nb::object g, double rstep, double rmin)
{
    QuantityType g0;
    const QuantityType& g1 = extractQuantityType(g, g0);
    QuantityType f = fftgtof(g1, rstep, rmin);
    nb::object fa = convertToNumPyArray(f);
    double rmaxpad = f.size() * rstep;
    double qstep = (rmaxpad > 0) ? (M_PI / rmaxpad) : 0.0;
    return nb::make_tuple(fa, qstep);
}

// pickling support ----------------------------------------------------------

template <class Super>
nb::tuple getstate_super(nb::object obj)
{
    // obtain C++ state without PDFEnvelopes
    nb::object envlps = obj.attr("envelopes");
    obj.attr("clearEnvelopes")();
    assert(nb::len(obj.attr("envelopes")) == 0);
    nb::tuple super_state = Super::getstate(obj);
    obj.attr("envelopes") = envlps;
    assert(nb::len(obj.attr("envelopes")) == nb::len(envlps));
    return nb::make_tuple(super_state);
}


nb::tuple getstate_common(nb::object obj)
{
    auto resolve_pwm = resolve_state_object<PeakWidthModel>;
    auto resolve_sft = resolve_state_object<ScatteringFactorTable>;
    nb::tuple rv = make_tuple(
            resolve_pwm(obj.attr("peakwidthmodel")),
            resolve_sft(obj.attr("scatteringfactortable")),
            obj.attr("envelopes")
            );
    return rv;
}


void setstate_common(nb::object obj, nb::tuple state, size_t& pos)
{
    assign_state_object(obj.attr("peakwidthmodel"), state[pos++]);
    assign_state_object(obj.attr("scatteringfactortable"), state[pos++]);
    assert(nb::len(obj.attr("envelopes")) == 0);
    obj.attr("envelopes") = nb::borrow<nb::object>(state[pos++]);
}


class DebyePDFCalculatorPickleSuite :
    public PairQuantityPickleSuite<DebyePDFCalculator, DICT_IGNORE>
{
    private:

        typedef PairQuantityPickleSuite<DebyePDFCalculator> Super;

    public:

        template <class C>
        static void bind(C& cls)
        {
            cls
                .def("__getstate__", getstate)
                .def("__setstate__", setstate)
                ;
        }
        

        static nb::tuple getstate(nb::object obj)
        {
            nb::tuple rv(
                    getstate_super<Super>(obj) +
                    getstate_common(obj)
                    );
            return rv;
        }


        static void setstate(nb::object obj, nb::tuple state)
        {
            ensure_tuple_length(state, 4);
            // restore the state using boost serialization
            nb::tuple st0(state[0]);
            Super::setstate(obj, st0);
            // other items are non-None only when restoring Python class
            size_t pos = 1;
            setstate_common(obj, state, pos);
        }
};


class PDFCalculatorPickleSuite :
    public PairQuantityPickleSuite<PDFCalculator, DICT_IGNORE>
{
    private:

        typedef PairQuantityPickleSuite<PDFCalculator> Super;

    public:

        template <class C>
        static void bind(C& cls)
        {
            cls
                .def("__getstate__", getstate)
                .def("__setstate__", setstate)
                ;
        }

        static nb::tuple getstate(nb::object obj)
        {
            nb::tuple mystate = make_tuple(
                    resolve_state_object<PeakProfile>(obj.attr("peakprofile")),
                    resolve_state_object<PDFBaseline>(obj.attr("baseline"))
                    );
            nb::tuple rv(
                    getstate_super<Super>(obj) +
                    getstate_common(obj) +
                    mystate
                    );
            return rv;
        }


        static void setstate(nb::object obj, nb::tuple state)
        {
            ensure_tuple_length(state, 6);
            // restore the state using boost serialization
            nb::tuple st0(state[0]);
            Super::setstate(obj, st0);
            // other items are non-None only when restoring Python class
            size_t pos = 1;
            setstate_common(obj, state, pos);
            assign_state_object(obj.attr("peakprofile"), state[pos++]);
            assign_state_object(obj.attr("baseline"), state[pos++]);
        }
};

}   // namespace nswrap_PDFCalculators

// Wrapper definition --------------------------------------------------------

void wrap_PDFCalculators(nb::module_& m)
{
    using namespace nswrap_PDFCalculators;

    // TODO: some types are flattened, we may need to add bindings manually
    // DebyePDFCalculator
    nb::class_<DebyePDFCalculator, PairQuantity>
            dbpdfc_class(m, "DebyePDFCalculator", doc_DebyePDFCalculator);
    wrap_PDFCommon<DebyePDFCalculator>(dbpdfc_class)
        .def(nb::init<>())
        .def("setOptimumQstep", &DebyePDFCalculator::setOptimumQstep,
                doc_DebyePDFCalculator_setOptimumQstep)
        .def("isOptimumQstep", &DebyePDFCalculator::isOptimumQstep,
                doc_DebyePDFCalculator_isOptimumQstep)
        ;
        DebyePDFCalculatorPickleSuite::bind(dbpdfc_class);

    // PDFCalculator
    nb::class_<PDFCalculator, PairQuantity>
        pdfc_class(m, "PDFCalculator", doc_PDFCalculator);
    wrap_PDFCommon<PDFCalculator>(pdfc_class)
        .def(nb::init<>())
        // PDF peak profile
        .def_prop_rw("peakprofile",
                getpeakprofile,
                setpeakprofile<PDFCalculator,PeakProfile>,
                doc_PDFCalculator_peakprofile)
        // PDF baseline
        .def_prop_rw("baseline",
                getbaseline,
                setbaseline<PDFCalculator,PDFBaseline>,
                doc_PDFCalculator_baseline)
        ;
        PDFCalculatorPickleSuite::bind(pdfc_class);

    // FFT functions
    m.def("fftftog", fftftog_array_step,
            nb::arg("f"), nb::arg("qstep"), nb::arg("qmin") = 0.0,
            doc_fftftog);
    m.def("fftgtof", fftgtof_array_step,
            nb::arg("g"), nb::arg("rstep"), nb::arg("rmin") = 0.0,
            doc_fftgtof);

}

}   // namespace srrealmodule

// End of file
