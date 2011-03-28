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

#include <diffpy/srreal/DebyePDFCalculator.hpp>
#include <diffpy/srreal/PDFCalculator.hpp>
#include <diffpy/srreal/PythonStructureAdapter.hpp>

#include "srreal_converters.hpp"
#include "srreal_pickling.hpp"

namespace srrealmodule {
namespace nswrap_PDFCalculators {

using namespace boost::python;
using namespace diffpy::srreal;

// docstrings

const char* doc_PDFCommon_pdf = "\
An array of PDF values in the form of G = 4*pi*r(rho - rho0) in A**-2.\n\
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

const char* doc_PDFCommon_addEnvelope = "FIXME\
";

const char* doc_PDFCommon_addEnvelopeByType = "\
";

const char* doc_PDFCommon_popEnvelope = "\
";

const char* doc_PDFCommon_popEnvelopeByType = "\
";

const char* doc_PDFCommon_getEnvelopeByType = "\
";

const char* doc_PDFCommon_usedEnvelopeTypes = "\
";

const char* doc_PDFCommon_clearEnvelopes = "\
";

const char* doc_PDFCommon_rdf = "\
";

const char* doc_PDFCalculator_getPeakProfile = "\
";

const char* doc_PDFCalculator_setPeakProfile = "\
";

const char* doc_PDFCalculator_setPeakProfileByType = "\
";

const char* doc_PDFCalculator_getBaseline = "\
";

const char* doc_PDFCalculator_setBaseline = "\
";

const char* doc_PDFCalculator_setBaselineByType = "\
";

// common wrappers

BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(getpkf_overloads,
        getPeakProfile, 0, 0)
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(getenvelopebytype_overloads,
        getEnvelopeByType, 1, 1)
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(getbaseline_overloads,
        getBaseline, 0, 0)

// DebyePDFCalculator and PDFCalculator wrappers

DECLARE_PYARRAY_METHOD_WRAPPER(getPDF, getPDF_asarray)
DECLARE_PYARRAY_METHOD_WRAPPER(getRDF, getRDF_asarray)
DECLARE_PYARRAY_METHOD_WRAPPER(getRgrid, getRgrid_asarray)
DECLARE_PYARRAY_METHOD_WRAPPER(getF, getF_asarray)
DECLARE_PYARRAY_METHOD_WRAPPER(getQgrid, getQgrid_asarray)
DECLARE_PYSET_METHOD_WRAPPER(usedEnvelopeTypes, usedEnvelopeTypes_asset)

}   // namespace nswrap_PDFCalculators

// Wrapper definition --------------------------------------------------------

void wrap_PDFCalculators()
{
    using namespace nswrap_PDFCalculators;

    // DebyePDFCalculator
    class_<DebyePDFCalculator,
        bases<PairQuantity, PeakWidthModelOwner, ScatteringFactorTableOwner>
            >("DebyePDFCalculator_ext")
        .add_property("pdf", getPDF_asarray<DebyePDFCalculator>,
                doc_PDFCommon_pdf)
        .add_property("rgrid", getRgrid_asarray<DebyePDFCalculator>,
                doc_PDFCommon_rgrid)
        .add_property("fq", getF_asarray<DebyePDFCalculator>,
                doc_PDFCommon_fq)
        .add_property("qgrid", getQgrid_asarray<DebyePDFCalculator>,
                doc_PDFCommon_qgrid)
        .def("setOptimumQstep", &DebyePDFCalculator::setOptimumQstep,
                doc_DebyePDFCalculator_setOptimumQstep)
        .def("isOptimumQstep", &DebyePDFCalculator::isOptimumQstep,
                doc_DebyePDFCalculator_isOptimumQstep)
        // PDF envelopes
        .def("addEnvelope", &DebyePDFCalculator::addEnvelope,
                doc_PDFCommon_addEnvelope)
        .def("addEnvelopeByType", &DebyePDFCalculator::addEnvelopeByType,
                doc_PDFCommon_addEnvelopeByType)
        .def("popEnvelope", &DebyePDFCalculator::popEnvelope,
                doc_PDFCommon_popEnvelope)
        .def("popEnvelopeByType", &DebyePDFCalculator::popEnvelopeByType,
                doc_PDFCommon_popEnvelopeByType)
        .def("getEnvelopeByType",
                (PDFEnvelopePtr(DebyePDFCalculator::*)(const std::string&)) NULL,
                getenvelopebytype_overloads(doc_PDFCommon_getEnvelopeByType))
        .def("usedEnvelopeTypes", usedEnvelopeTypes_asset<DebyePDFCalculator>,
                doc_PDFCommon_usedEnvelopeTypes)
        .def("clearEnvelopes", &DebyePDFCalculator::clearEnvelopes,
                doc_PDFCommon_clearEnvelopes)
        .def_pickle(SerializationPickleSuite<DebyePDFCalculator>())
        ;

    // PDFCalculator
    class_<PDFCalculator,
        bases<PairQuantity, PeakWidthModelOwner, ScatteringFactorTableOwner>
        >("PDFCalculator_ext")
        .add_property("pdf", getPDF_asarray<PDFCalculator>,
                doc_PDFCommon_pdf)
        .add_property("rdf", getRDF_asarray<PDFCalculator>,
                doc_PDFCommon_rdf)
        .add_property("rgrid", getRgrid_asarray<PDFCalculator>,
                doc_PDFCommon_rgrid)
        .add_property("fq", getF_asarray<PDFCalculator>,
                doc_PDFCommon_fq)
        .add_property("qgrid", getQgrid_asarray<PDFCalculator>,
                doc_PDFCommon_qgrid)
        // PDF peak profile
        .def("getPeakProfile",
                (PeakProfilePtr(PDFCalculator::*)()) NULL,
                getpkf_overloads(doc_PDFCalculator_getPeakProfile))
        .def("setPeakProfile", &PDFCalculator::setPeakProfile,
                doc_PDFCalculator_setPeakProfile)
        .def("setPeakProfileByType", &PDFCalculator::setPeakProfileByType,
                doc_PDFCalculator_setPeakProfileByType)
        // PDF baseline
        .def("getBaseline",
                (PDFBaselinePtr(PDFCalculator::*)()) NULL,
                getbaseline_overloads(doc_PDFCalculator_getBaseline))
        .def("setBaseline", &PDFCalculator::setBaseline,
                doc_PDFCalculator_setBaseline)
        .def("setBaselineByType", &PDFCalculator::setBaselineByType,
                doc_PDFCalculator_setBaselineByType)
        // PDF envelopes
        .def("addEnvelope", &PDFCalculator::addEnvelope,
                doc_PDFCommon_addEnvelope)
        .def("addEnvelopeByType", &PDFCalculator::addEnvelopeByType,
                doc_PDFCommon_addEnvelopeByType)
        .def("popEnvelope", &PDFCalculator::popEnvelope,
                doc_PDFCommon_popEnvelope)
        .def("popEnvelopeByType", &PDFCalculator::popEnvelopeByType,
                doc_PDFCommon_popEnvelopeByType)
        .def("getEnvelopeByType",
                (PDFEnvelopePtr(PDFCalculator::*)(const std::string&)) NULL,
                getenvelopebytype_overloads(doc_PDFCommon_getEnvelopeByType))
        .def("usedEnvelopeTypes", usedEnvelopeTypes_asset<PDFCalculator>,
                doc_PDFCommon_usedEnvelopeTypes)
        .def("clearEnvelopes", &PDFCalculator::clearEnvelopes,
                doc_PDFCommon_clearEnvelopes)
        .def_pickle(SerializationPickleSuite<PDFCalculator>())
        ;
}

}   // namespace srrealmodule

// End of file
