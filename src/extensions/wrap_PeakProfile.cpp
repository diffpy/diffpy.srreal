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
* Bindings to the PeakProfile class.  The business methods can be overridden
* from Python to create custom peak profiles.  This may be however quite slow.
*
*****************************************************************************/

#include <nanobind/nanobind.h>
#include <nanobind/trampoline.h>

#include <string>

#include <diffpy/srreal/PeakProfile.hpp>
#include <diffpy/srreal/GaussianProfile.hpp>
#include <diffpy/srreal/CroppedGaussianProfile.hpp>

#include "srreal_converters.hpp"
#include "srreal_pickling.hpp"
#include "srreal_registry.hpp"

namespace srrealmodule {
namespace nswrap_PeakProfile {

using namespace diffpy::srreal;

// docstrings ----------------------------------------------------------------

const char* doc_PeakProfile = "\
Base class and registry for PDF peak profile functions.\n\
The peak profile functions must be centered at 0 and their integrated\n\
area must be 1.\n\
";

const char* doc_PeakProfile___call__ = "\
Evaluate peak profile function for a given position and peak width.\n\
\n\
x    -- coordinate where the profile function is evaluated.\n\
        x is always relative to the peak center, i.e., the profile function\n\
        must be centered at 0.\n\
fwhm -- Full Width at Half Maximum of the profile function.\n\
\n\
Return float.\n\
";

const char* doc_PeakProfile_xboundlo = "\
Lower x-bound where profile function becomes smaller than precision.\n\
The bound is evaluated relative to profile maximum, i.e., for each x below\n\
xboundlo the profile function must be smaller than (peakprecision * maximum).\n\
\n\
fwhm -- Full Width at Half Maximum of the profile function\n\
\n\
Return float.  See also the peakprecision property.\n\
";

const char* doc_PeakProfile_xboundhi = "\
Upper x-bound where profile function becomes smaller than precision.\n\
The bound is evaluated relative to profile maximum, i.e., for each x above\n\
xboundhi the profile function must be smaller than (peakprecision * maximum).\n\
\n\
fwhm -- Full Width at Half Maximum of the profile function\n\
\n\
Return float.  See also the peakprecision property.\n\
";

const char* doc_PeakProfile_ticker = "\
Return EventTicker that marks last modification time of this object.\n\
This ticker object is used in fast PDF update, to check if PeakProfile\n\
has changed since the last calculation.  The ticker.click() method needs\n\
to be therefore called after every change in PeakProfile configuration.\n\
\n\
Return EventTicker object.\n\
This method can be overridden in a Python-derived class.\n\
";

const char* doc_PeakProfile__registerThisType = "\
Add this class to the global registry of PeakProfile types.\n\
\n\
This method must be called once after definition of the derived\n\
class to support pickling and the createByType factory.\n\
";

const char* doc_GaussianProfile = "\
Gaussian profile function.\n\
";

const char* doc_CroppedGaussianProfile = "\
Gaussian function cropped to zero at its crossing with peakprecision.\n\
\n\
The profile is also rescaled to keep the integrated area of 1.\n\
";

// wrappers ------------------------------------------------------------------

// Support for override of PeakProfile methods from Python.

class PeakProfileWrap :
    public PeakProfile,
    public PythonTrampolineTag
{
    public:

        NB_TRAMPOLINE(PeakProfile, 7);

        // HasClassRegistry methods

        PeakProfilePtr create() const override
        {
            nb::gil_scoped_acquire gil;
            nb::detail::ticket ticket(nb_trampoline, "create", true);

            if (!ticket.key.is_valid()) 
            {
                    throw nb::type_error(
                        "pure virtual method PeakProfile.create() called"
                    );
            }

            nb::object rv = nb_trampoline.base().attr(ticket.key)();
            return mconfigurator.fetch(rv);
        }

        PeakProfilePtr clone() const override
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
                    "pure virtual method PeakProfile.type() called"
                );
            }

            nb::object tp = nb_trampoline.base().attr(ticket.key)();
            mtype = nb::cast<std::string>(tp);
            return mtype;
        }

        // own methods

        double operator()(double x, double fwhm) const override
        {
            NB_OVERRIDE_PURE_NAME("__call__", operator(), x, fwhm);
        }

        double xboundlo(double fwhm) const override
        {
            NB_OVERRIDE_PURE(xboundlo, fwhm);
        }

        double xboundhi(double fwhm) const override
        {
            NB_OVERRIDE_PURE(xboundhi, fwhm);
        }

        // Support for ticker override from Python.

        diffpy::eventticker::EventTicker& ticker() const override
        {
            using diffpy::eventticker::EventTicker;
            nb::gil_scoped_acquire gil;
            nb::detail::ticket ticket(nb_trampoline, "ticker", false);

            if (ticket.key.is_valid()) 
            {
                nb::object ptic = nb_trampoline.base().attr(ticket.key)();
                return nb::cast<EventTicker&>(ptic);
            }

            return this->default_ticker();
        }

        diffpy::eventticker::EventTicker& default_ticker() const
        {
            return this->PeakProfile::ticker();
        }

    protected:

        // HasClassRegistry method

        void setupRegisteredObject(PeakProfilePtr p) const override
        {
            mconfigurator.setup(p);
        }

    private:

        mutable std::string mtype;
        wrapper_registry_configurator<PeakProfile> mconfigurator;

        // serialization
        friend class boost::serialization::access;
        template<class Archive>
            void serialize(Archive& ar, const unsigned int version)
        {
            using boost::serialization::base_object;
            ar & base_object<PeakProfile>(*this);
        }

};  // class PeakProfileWrap

}   // namespace nswrap_PeakProfile

// Wrapper definition --------------------------------------------------------

void wrap_PeakProfile(nb::module_& m)
{
    using namespace nswrap_PeakProfile;
    using diffpy::Attributes;

    nb::class_<PeakProfile, Attributes, PeakProfileWrap>
        peakprofile(m, "PeakProfile", nb::dynamic_attr(), doc_PeakProfile);
    wrap_registry_methods(peakprofile)
        .def(nb::init<>())
        .def("__call__", &PeakProfile::operator(),
                nb::arg("x"), nb::arg("fwhm"), doc_PeakProfile___call__)
        .def("xboundlo", &PeakProfile::xboundlo,
                nb::arg("fwhm"), doc_PeakProfile_xboundlo)
        .def("xboundhi", &PeakProfile::xboundhi,
                nb::arg("fwhm"), doc_PeakProfile_xboundhi)
        .def("ticker",
                &PeakProfile::ticker,
                nb::rv_policy::reference_internal,
                doc_PeakProfile_ticker)
        ;

    SerializationPickleSuite<
        PeakProfile,
        DICT_PICKLE,
        PeakProfileWrap>::bind(peakprofile);

    nb::class_<GaussianProfile, PeakProfile> gaussianprofile(m,
            "GaussianProfile", doc_GaussianProfile);
    gaussianprofile
        .def(nb::init<>())
        ;
        SerializationPickleSuite<GaussianProfile, DICT_IGNORE>::bind(gaussianprofile);

    nb::class_<CroppedGaussianProfile, GaussianProfile> croppedgaussianprofile(m,
            "CroppedGaussianProfile", doc_CroppedGaussianProfile);
    croppedgaussianprofile
        .def(nb::init<>())
        ;
        SerializationPickleSuite<CroppedGaussianProfile, DICT_IGNORE>::bind(croppedgaussianprofile);

}

}   // namespace srrealmodule

// Serialization -------------------------------------------------------------

BOOST_CLASS_EXPORT(srrealmodule::nswrap_PeakProfile::PeakProfileWrap)

// End of file
