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
* Bindings to DebyePDFCalculator and PDFCalculator classes.
*
* $Id$
*
*****************************************************************************/

#include <boost/python.hpp>
#include <boost/python/stl_iterator.hpp>

#include <diffpy/srreal/DebyePDFCalculator.hpp>
#include <diffpy/srreal/PDFCalculator.hpp>
#include <diffpy/srreal/PythonStructureAdapter.hpp>

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
envlp    -- instance of PDFEnvelope that defines the scaling function.\n\
\n\
No return value.\n\
";

const char* doc_PDFCommon_addEnvelopeByType = "\
Add a scaling PDFEnvelope object of the specified string type.\n\
This replaces any existing PDFEnvelope of the same type.\n\
\n\
tp       -- string identifier of a registered PDFEnvelope type.\n\
            Use PDFEnvelope.getRegisteredTypes for the allowed values.\n\
\n\
No return value.\n\
";

const char* doc_PDFCommon_popEnvelope = "\
Remove a PDFEnvelope object from an internal list of scaling functions.\n\
\n\
envlp    -- instance of PDFEnvelope object to be removed.\n\
            No action if envlp is not present in the calculator.\n\
            See the 'envelopes' attribute for a list of active\n\
            PDFEnvelope instances.\n\
\n\
No return value.\n\
";

const char* doc_PDFCommon_popEnvelopeByType = "\
Remove a PDFEnvelope of the specified type from the internal list\n\
of scaling functions.\n\
\n\
tp       -- string identifier of a registered PDFEnvelope type.\n\
            No action if the envelope type tp does not exist.\n\
            See the 'usedenvelopetypes' attribute for a list\n\
            of active PDFEnvelope types.\n\
\n\
No return value.\n\
";

const char* doc_PDFCommon_getEnvelopeByType = "\
Retrieve an active PDFEnvelope object by its string type.\n\
\n\
tp       -- string type of a PDFEnvelope object that is present\n\
            in the calculator.\n\
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

const char* doc_PDFCalculator_getPeakProfile = "\
FIXME\n\
";

const char* doc_PDFCalculator_setPeakProfile = "\
FIXME\n\
";

const char* doc_PDFCalculator_setPeakProfileByType = "\
FIXME\n\
";

const char* doc_PDFCalculator_baseline = "\
Instance of PDFBaseline that calculates unscaled baseline at r.\n\
";

const char* doc_PDFCalculator_setBaselineByType = "\
FIXME\n\
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

BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(getpkf_overloads,
        getPeakProfile, 0, 0)
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(getenvelopebytype_overloads,
        getEnvelopeByType, 1, 1)

DECLARE_PYARRAY_METHOD_WRAPPER(getPDF, getPDF_asarray)
DECLARE_PYARRAY_METHOD_WRAPPER(getRDF, getRDF_asarray)
DECLARE_PYARRAY_METHOD_WRAPPER(getRgrid, getRgrid_asarray)
DECLARE_PYARRAY_METHOD_WRAPPER(getF, getF_asarray)
DECLARE_PYARRAY_METHOD_WRAPPER(getQgrid, getQgrid_asarray)
DECLARE_PYLIST_METHOD_WRAPPER(usedEnvelopeTypes, usedEnvelopeTypes_aslist)

// wrappers for the baseline property

PDFBaselinePtr getbaseline(const PDFCalculator& obj)
{
    return obj.getBaseline();
}

void setbaseline(PDFCalculator& obj, PDFBaselinePtr bl)
{
    obj.setBaseline(bl);
}

// wrappers for the envelopes property

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
    stl_input_iterator<PDFEnvelopePtr> begin(evps), end;
    // first check if all evps items can be converted to PDFEnvelopePtr
    std::list<PDFEnvelopePtr> elst(begin, end);
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
        .def("addEnvelope", &W::addEnvelope,
                bp::arg("envlp"), doc_PDFCommon_addEnvelope)
        .def("addEnvelopeByType", &W::addEnvelopeByType,
                bp::arg("tp"), doc_PDFCommon_addEnvelopeByType)
        .def("popEnvelope", &W::popEnvelope,
                doc_PDFCommon_popEnvelope)
        .def("popEnvelopeByType", &W::popEnvelopeByType,
                bp::arg("tp"), doc_PDFCommon_popEnvelopeByType)
        .def("getEnvelopeByType",
                (PDFEnvelopePtr(W::*)(const std::string&)) NULL,
                getenvelopebytype_overloads(
                    bp::arg("tp"), doc_PDFCommon_getEnvelopeByType))
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
        .def_pickle(SerializationPickleSuite<DebyePDFCalculator>())
        ;

    // PDFCalculator
    class_<PDFCalculator,
        bases<PairQuantity, PeakWidthModelOwner, ScatteringFactorTableOwner> >
        pdfc_class("PDFCalculator", doc_PDFCalculator);
    wrap_PDFCommon(pdfc_class)
        // PDF peak profile
        .def("getPeakProfile",
                (PeakProfilePtr(PDFCalculator::*)()) NULL,
                getpkf_overloads(doc_PDFCalculator_getPeakProfile))
        .def("setPeakProfile", &PDFCalculator::setPeakProfile,
                doc_PDFCalculator_setPeakProfile)
        .def("setPeakProfileByType", &PDFCalculator::setPeakProfileByType,
                doc_PDFCalculator_setPeakProfileByType)
        // PDF baseline
        .add_property("baseline", getbaseline, setbaseline,
                doc_PDFCalculator_baseline)
        .def("setBaselineByType", &PDFCalculator::setBaselineByType,
                doc_PDFCalculator_setBaselineByType)
        .def_pickle(SerializationPickleSuite<PDFCalculator>())
        ;

    // FFT functions
    def("fftftog", fftftog_array_step,
            (bp::arg("f"), bp::arg("qstep"), bp::arg("qmin")=0.0),
            doc_fftftog);
    def("fftgtof", fftgtof_array_step,
            (bp::arg("g"), bp::arg("rstep"), bp::arg("rmin")=0.0),
            doc_fftgtof);

    // inject pickling methods for PDFBaseline and PDFEnvelope classes
    import("diffpy.srreal.pdfcalculator");
}

}   // namespace srrealmodule

// End of file
