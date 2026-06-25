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
* Bindings to the PeakWidthModel class.  The business methods can be overridden
* from Python to create custom peak widths.  This may be however quite slow.
*
*****************************************************************************/

#include <nanobind/nanobind.h>
#include <nanobind/trampoline.h>

#include <string>

#include <diffpy/srreal/PeakWidthModel.hpp>
#include <diffpy/srreal/ConstantPeakWidth.hpp>
#include <diffpy/srreal/DebyeWallerPeakWidth.hpp>
#include <diffpy/srreal/JeongPeakWidth.hpp>
#include <diffpy/serialization.hpp>

#include "srreal_converters.hpp"
#include "srreal_pickling.hpp"
#include "srreal_registry.hpp"

namespace nb = nanobind;

namespace srrealmodule {
namespace nswrap_PeakWidthModel {

using namespace diffpy::srreal;

// docstrings ----------------------------------------------------------------

const char* doc_PeakWidthModel = "\
Base class for functors that calculate the PDF peak widths.\n\
Peak width is defined as full width at half maximum (FWHM).\n\
";

const char* doc_PeakWidthModel_calculate = "\
Calculate the FWHM peak width for the specified bond.\n\
\n\
bnds -- instance of BaseBondGenerator with the current bond data.\n\
\n\
Return float.\n\
This method must be overridden in a derived class.\n\
";

const char* doc_PeakWidthModel_maxWidth = "\
Return maximum peak width for the specified structure and distance range\n\
\n\
stru -- StructureAdapter object or an object convertible to StructureAdapter.\n\
rmin -- lower bound for the PDF calculation\n\
rmax -- upper bound for the PDF calculation\n\
        isotropically vibrating atoms this\n\
\n\
Return float.\n\
";

const char* doc_PeakWidthModel_ticker = "\
Return EventTicker that marks last modification time of this object.\n\
This ticker object is used in fast PDF update, to check if PeakWidthModel\n\
has changed since the last calculation.  The ticker.click() method needs\n\
to be therefore called after every change in PeakWidthModel configuration.\n\
\n\
Return EventTicker object.\n\
This method can be overridden in a Python-derived class.\n\
";

const char* doc_ConstantPeakWidth = "\
Peak width model returning constant width.\n\
";

const char* doc_DebyeWallerPeakWidth = "\
Calculate PDF peak width from a Debye-Waller model of atom vibrations.\n\
This returns mean-square displacement of atoms in the pair along their\n\
bond scaled as FWHM.  The atom displacements are assumed independent.\n\
";

const char* doc_JeongPeakWidth = "\
Calculate PDF peak width assuming I.-K. Jeong model of correlated motion.\n\
This returns mean-square displacement of atoms in the pair along their\n\
bond corrected for correlated motion and data noise at high-Q by a scaling\n\
factor  sqrt(1 - delta1/r - delta2/r**2 + (qbroad * r)**2).\n\
";

const char* doc_PeakWidthModelOwner = "\
Base class for classes that own PeakWidthModel instance.\n\
";

const char* doc_PeakWidthModelOwner_peakwidthmodel = "\
PeakWidthModel object used for calculating the FWHM of the PDF peaks.\n\
This attribute can be assigned either a PeakWidthModel-derived object\n\
or a string name of a registered PeakWidthModel class.\n\
Use PeakWidthModel.getRegisteredTypes() for a set of registered names.\n\
";

// wrappers ------------------------------------------------------------------

double maxwidthwithpystructure(const PeakWidthModel& pwm,
        nb::object stru, double rmin, double rmax)
{
    StructureAdapterPtr adpt = createStructureAdapter(stru);
    double rv = pwm.maxWidth(adpt, rmin, rmax);
    return rv;
}

// wrappers for the peakwidthmodel property

PeakWidthModelPtr getpwmodel(PeakWidthModelOwner& obj)
{
    return obj.getPeakWidthModel();
}

DECLARE_BYTYPE_SETTER_WRAPPER(setPeakWidthModel, setpwmodel)

// Support for Python override of the PeakWidthModel methods.

class PeakWidthModelWrap :
    public PeakWidthModel
{
    public:

        NB_TRAMPOLINE(PeakWidthModel, 6);

        // HasClassRegistry methods

        PeakWidthModelPtr create() const override
        {
            nb::gil_scoped_acquire gil;
            nb::detail::ticket ticket(nb_trampoline, "create", true);

            if (!ticket.key.is_valid()) 
            {
                throw nb::type_error(
                    "pure virtual method PeakWidthModel.create() called"
                );
            }

            nb::object rv = nb_trampoline.base().attr(ticket.key)();
            return mconfigurator.fetch(rv);
        }

        PeakWidthModelPtr clone() const override
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
                    "pure virtual method PeakWidthModel.type() called"
                );
            }

            nb::object tp = nb_trampoline.base().attr(ticket.key)();
            mtype = nb::cast<std::string>(tp);
            return mtype;
        }

        // own methods

        double calculate(const BaseBondGenerator& bnds) const override
        {
            NB_OVERRIDE_PURE(calculate, bnds);
        }

        double maxWidth(StructureAdapterPtr stru,
                double rmin, double rmax) const override
        {
            NB_OVERRIDE_PURE(maxWidth, stru, rmin, rmax);
        }

        // Make the ticker method overridable from Python

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
            return this->PeakWidthModel::ticker();
        }

    protected:

        // HasClassRegistry method

        void setupRegisteredObject(PeakWidthModelPtr p) const override
        {
            mconfigurator.setup(p);
        }

    private:

        mutable std::string mtype;
        wrapper_registry_configurator<PeakWidthModel> mconfigurator;

        // serialization
        friend class boost::serialization::access;
        template<class Archive>
            void serialize(Archive& ar, const unsigned int version)
        {
            using boost::serialization::base_object;
            ar & base_object<PeakWidthModel>(*this);
        }

};  // class PeakWidthModelWrap

}   // namespace nswrap_PeakWidthModel

// Wrapper definition --------------------------------------------------------

void wrap_PeakWidthModel(nb::module_& m)
{
    using namespace nswrap_PeakWidthModel;
    using diffpy::Attributes;

    nb::class_<PeakWidthModel, Attributes, PeakWidthModelWrap>
        peakwidthmodel(m, "PeakWidthModel", nb::dynamic_attr(), doc_PeakWidthModel);
    wrap_registry_methods(peakwidthmodel)
        .def(nb::init<>())
        .def("calculate",
                &PeakWidthModel::calculate,
                nb::arg("bnds"),
                doc_PeakWidthModel_calculate)
        .def("maxWidth",
                &PeakWidthModel::maxWidth,
                nb::arg("stru"), nb::arg("rmin"), nb::arg("rmax"),
                doc_PeakWidthModel_maxWidth)
        .def("maxWidth",
                maxwidthwithpystructure,
                nb::arg("stru"), nb::arg("rmin"), nb::arg("rmax"))
        .def("ticker",
                &PeakWidthModel::ticker,
                nb::rv_policy::reference_internal,
                doc_PeakWidthModel_ticker)
        ;
        SerializationPickleSuite<PeakWidthModel, DICT_PICKLE>::bind(peakwidthmodel);

    nb::class_<ConstantPeakWidth, PeakWidthModel> constantpeakwidth(m,
            "ConstantPeakWidth", doc_ConstantPeakWidth);
    constantpeakwidth
        .def(nb::init<>())
        ;
        SerializationPickleSuite<ConstantPeakWidth, DICT_IGNORE>::bind(constantpeakwidth);
    
    nb::class_<DebyeWallerPeakWidth, PeakWidthModel> debywallerpeakwidth(m,
            "DebyeWallerPeakWidth", doc_DebyeWallerPeakWidth);
    debywallerpeakwidth
        .def(nb::init<>())
        ;
        SerializationPickleSuite<DebyeWallerPeakWidth, DICT_IGNORE>::bind(debywallerpeakwidth);

    nb::class_<JeongPeakWidth, DebyeWallerPeakWidth> jeongpeakwidth(m,
            "JeongPeakWidth", doc_JeongPeakWidth);
    jeongpeakwidth
        .def(nb::init<>())
        ;
        SerializationPickleSuite<JeongPeakWidth, DICT_IGNORE>::bind(jeongpeakwidth);

    nb::class_<PeakWidthModelOwner>(m, "PeakWidthModelOwner", doc_PeakWidthModelOwner)
        .def_prop_rw("peakwidthmodel",
                getpwmodel,
                setpwmodel<PeakWidthModelOwner,PeakWidthModel>,
                doc_PeakWidthModelOwner_peakwidthmodel)
        ;
}

}   // namespace srrealmodule

// Serialization -------------------------------------------------------------

BOOST_CLASS_EXPORT(srrealmodule::nswrap_PeakWidthModel::PeakWidthModelWrap)

// End of file
