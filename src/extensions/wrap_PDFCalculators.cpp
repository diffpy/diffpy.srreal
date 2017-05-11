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

#include <boost/python/class.hpp>
#include <boost/python/def.hpp>
#include <boost/python/stl_iterator.hpp>

#include <diffpy/srreal/DebyePDFCalculator.hpp>
#include <diffpy/srreal/PDFCalculator.hpp>

#include "srreal_converters.hpp"
#include "srreal_pickling.hpp"

namespace srrealmodule {
namespace nswrap_PDFCalculators {

using namespace boost::python;
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

PDFEnvelopePtr pyobjtoenvelope(object evp)
{
    extract<std::string> tp(evp);
    if (tp.check())  return PDFEnvelope::createByType(tp());
    PDFEnvelopePtr rv = extract<PDFEnvelopePtr>(evp);
    return rv;
}


template <class T>
tuple getenvelopes(T& obj)
{
    std::set<std::string> etps = obj.usedEnvelopeTypes();
    std::set<std::string>::const_iterator tpi;
    list elst;
    for (tpi = etps.begin(); tpi != etps.end(); ++tpi)
    {
        elst.append(obj.getEnvelopeByType(*tpi));
    }
    tuple rv(elst);
    return rv;
}

template <class T>
void setenvelopes(T& obj, object evps)
{
    stl_input_iterator<object> eit(evps), end;
    // first check if all evps items can be converted to PDFEnvelopePtr
    std::list<PDFEnvelopePtr> elst;
    for (; eit != end; ++eit)  elst.push_back(pyobjtoenvelope(*eit));
    // if that worked, overwrite the original envelopes
    obj.clearEnvelopes();
    std::list<PDFEnvelopePtr>::iterator eii = elst.begin();
    for (; eii != elst.end(); ++eii)  obj.addEnvelope(*eii);
}

// wrapper for the usedenvelopetypes

template <class T>
tuple getusedenvelopetypes(T& obj)
{
    tuple rv(usedEnvelopeTypes_aslist<T>(obj));
    return rv;
}

template <class T>
void addenvelope(T& obj, object evp)
{
    PDFEnvelopePtr e = pyobjtoenvelope(evp);
    obj.addEnvelope(e);
}

template <class T>
void popenvelope(T& obj, object evp)
{
    extract<std::string> tp(evp);
    if (tp.check())  obj.popEnvelopeByType(tp());
    else
    {
        PDFEnvelopePtr e = extract<PDFEnvelopePtr>(evp);
        obj.popEnvelope(e);
    }
}

template <class T>
PDFEnvelopePtr getoneenvelope(T& obj, const std::string& tp)
{
    return obj.getEnvelopeByType(tp);
}


// wrap shared methods and attributes of PDFCalculators

template <class C>
C& wrap_PDFCommon(C& boostpythonclass)
{
    namespace bp = boost::python;
    typedef typename C::wrapped_type W;
    boostpythonclass
        // result vectors
        .add_property("pdf", getPDF_asarray<W>,
                doc_PDFCommon_pdf)
        .add_property("rdf", getRDF_asarray<W>,
                doc_PDFCommon_rdf)
        .add_property("rgrid", getRgrid_asarray<W>,
                doc_PDFCommon_rgrid)
        .add_property("fq", getF_asarray<W>,
                doc_PDFCommon_fq)
        .add_property("qgrid", getQgrid_asarray<W>,
                doc_PDFCommon_qgrid)
        // PDF envelopes
        .add_property("envelopes",
                getenvelopes<W>, setenvelopes<W>,
                doc_PDFCommon_envelopes)
        .add_property("usedenvelopetypes",
                getusedenvelopetypes<W>,
                doc_PDFCommon_usedenvelopetypes)
        .def("addEnvelope", addenvelope<W>,
                bp::arg("envlp"), doc_PDFCommon_addEnvelope)
        .def("popEnvelope", popenvelope<W>,
                doc_PDFCommon_popEnvelope)
        .def("getEnvelope", getoneenvelope<W>,
                bp::arg("tp"), doc_PDFCommon_getEnvelope)
        .def("clearEnvelopes", &W::clearEnvelopes,
                doc_PDFCommon_clearEnvelopes)
        ;
    return boostpythonclass;
}

// local helper for converting python object to a quantity type

tuple fftftog_array_step(object f, double qstep, double qmin)
{
    QuantityType f0;
    const QuantityType& f1 = extractQuantityType(f, f0);
    QuantityType g = fftftog(f1, qstep, qmin);
    object ga = convertToNumPyArray(g);
    double qmaxpad = g.size() * qstep;
    double rstep = (qmaxpad > 0) ? (M_PI / qmaxpad) : 0.0;
    return make_tuple(ga, rstep);
}


tuple fftgtof_array_step(object g, double rstep, double rmin)
{
    QuantityType g0;
    const QuantityType& g1 = extractQuantityType(g, g0);
    QuantityType f = fftgtof(g1, rstep, rmin);
    object fa = convertToNumPyArray(f);
    double rmaxpad = f.size() * rstep;
    double qstep = (rmaxpad > 0) ? (M_PI / rmaxpad) : 0.0;
    return make_tuple(fa, qstep);
}

}   // namespace nswrap_PDFCalculators

// Wrapper definition --------------------------------------------------------

void wrap_PDFCalculators()
{
    using namespace nswrap_PDFCalculators;
    namespace bp = boost::python;

    // DebyePDFCalculator
    class_<DebyePDFCalculator,
        bases<PairQuantity, PeakWidthModelOwner, ScatteringFactorTableOwner> >
            dbpdfc_class("DebyePDFCalculator", doc_DebyePDFCalculator);
    wrap_PDFCommon(dbpdfc_class)
        .def("setOptimumQstep", &DebyePDFCalculator::setOptimumQstep,
                doc_DebyePDFCalculator_setOptimumQstep)
        .def("isOptimumQstep", &DebyePDFCalculator::isOptimumQstep,
                doc_DebyePDFCalculator_isOptimumQstep)
        .def_pickle(PairQuantityPickleSuite<DebyePDFCalculator>())
        ;

    // PDFCalculator
    class_<PDFCalculator,
        bases<PairQuantity, PeakWidthModelOwner, ScatteringFactorTableOwner> >
        pdfc_class("PDFCalculator", doc_PDFCalculator);
    wrap_PDFCommon(pdfc_class)
        // PDF peak profile
        .add_property("peakprofile",
                getpeakprofile,
                setpeakprofile<PDFCalculator,PeakProfile>,
                doc_PDFCalculator_peakprofile)
        // PDF baseline
        .add_property("baseline",
                getbaseline,
                setbaseline<PDFCalculator,PDFBaseline>,
                doc_PDFCalculator_baseline)
        .def_pickle(PairQuantityPickleSuite<PDFCalculator>())
        ;

    // FFT functions
    def("fftftog", fftftog_array_step,
            (bp::arg("f"), bp::arg("qstep"), bp::arg("qmin")=0.0),
            doc_fftftog);
    def("fftgtof", fftgtof_array_step,
            (bp::arg("g"), bp::arg("rstep"), bp::arg("rmin")=0.0),
            doc_fftgtof);

}

}   // namespace srrealmodule

// End of file
