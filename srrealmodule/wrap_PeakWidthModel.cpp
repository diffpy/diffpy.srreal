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

#include <boost/python/class.hpp>
#include <boost/python/def.hpp>
#include <boost/python/copy_const_reference.hpp>
#include <boost/python/register_ptr_to_python.hpp>

#include <string>

#include <diffpy/srreal/PeakWidthModel.hpp>
#include <diffpy/srreal/ConstantPeakWidth.hpp>
#include <diffpy/srreal/DebyeWallerPeakWidth.hpp>
#include <diffpy/srreal/JeongPeakWidth.hpp>
#include <diffpy/serialization.hpp>

#include "srreal_converters.hpp"
#include "srreal_registry.hpp"

namespace srrealmodule {
namespace nswrap_PeakWidthModel {

using namespace boost;
using namespace boost::python;
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
        python::object stru, double rmin, double rmax)
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
    public PeakWidthModel,
    public wrapper_srreal<PeakWidthModel>
{
    public:

        // HasClassRegistry methods

        PeakWidthModelPtr create() const
        {
            object rv = this->get_pure_virtual_override("create")();
            return mconfigurator.fetch(rv);
        }

        PeakWidthModelPtr clone() const
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

        double calculate(const BaseBondGenerator& bnds) const
        {
            return this->get_pure_virtual_override("calculate")(ptr(&bnds));
        }

        double maxWidth(StructureAdapterPtr stru,
                double rmin, double rmax) const
        {
            override f = this->get_pure_virtual_override("maxWidth");
            return f(stru, rmin, rmax);
        }

        // Make the ticker method overridable from Python

        diffpy::eventticker::EventTicker& ticker() const
        {
            using diffpy::eventticker::EventTicker;
            override f = this->get_override("ticker");
            if (f)
            {
                // avoid "dangling reference error" when used from C++
                python::object ptic = f();
                return python::extract<EventTicker&>(ptic);
            }
            return this->default_ticker();
        }

        diffpy::eventticker::EventTicker& default_ticker() const
        {
            return this->PeakWidthModel::ticker();
        }

    protected:

        // HasClassRegistry method

        void setupRegisteredObject(PeakWidthModelPtr p) const
        {
            mconfigurator.setup(p);
        }

    private:

        mutable std::string mtype;
        wrapper_registry_configurator<PeakWidthModel> mconfigurator;

};  // class PeakWidthModelWrap


std::string peakwidthmodel_tostring(PeakWidthModelPtr obj)
{
    return diffpy::serialization_tostring(obj);
}


PeakWidthModelPtr peakwidthmodel_fromstring(const std::string& content)
{
    PeakWidthModelPtr rv;
    diffpy::serialization_fromstring(rv, content);
    return rv;
}


}   // namespace nswrap_PeakWidthModel

// Wrapper definition --------------------------------------------------------

void wrap_PeakWidthModel()
{
    namespace bp = boost::python;
    using namespace nswrap_PeakWidthModel;
    using diffpy::Attributes;

    class_<PeakWidthModelWrap, bases<Attributes>, noncopyable>
        peakwidthmodel("PeakWidthModel", doc_PeakWidthModel);
    wrap_registry_methods(peakwidthmodel)
        .def("calculate",
                &PeakWidthModel::calculate,
                bp::arg("bnds"),
                doc_PeakWidthModel_calculate)
        .def("maxWidth",
                &PeakWidthModel::maxWidth,
                (bp::arg("stru"), bp::arg("rmin"), bp::arg("rmax")),
                doc_PeakWidthModel_maxWidth)
        .def("maxWidth",
                maxwidthwithpystructure,
                (bp::arg("stru"), bp::arg("rmin"), bp::arg("rmax")))
        .def("maxWidth",
                &PeakWidthModel::maxWidth,
                (bp::arg("stru"), bp::arg("rmin"), bp::arg("rmax")),
                doc_PeakWidthModel_maxWidth)
        .def("ticker",
                &PeakWidthModel::ticker,
                &PeakWidthModelWrap::default_ticker,
                return_internal_reference<>(),
                doc_PeakWidthModel_ticker)
        .enable_pickling()
        ;

    register_ptr_to_python<PeakWidthModelPtr>();

    class_<ConstantPeakWidth, bases<PeakWidthModel> >(
            "ConstantPeakWidth", doc_ConstantPeakWidth)
        ;

    class_<DebyeWallerPeakWidth, bases<PeakWidthModel> >(
            "DebyeWallerPeakWidth", doc_DebyeWallerPeakWidth)
        ;

    class_<JeongPeakWidth, bases<DebyeWallerPeakWidth> >(
            "JeongPeakWidth", doc_JeongPeakWidth)
        ;

    // pickling support functions
    def("_PeakWidthModel_tostring", peakwidthmodel_tostring);
    def("_PeakWidthModel_fromstring", peakwidthmodel_fromstring);

    class_<PeakWidthModelOwner>("PeakWidthModelOwner", doc_PeakWidthModelOwner)
        .add_property("peakwidthmodel",
                getpwmodel,
                setpwmodel<PeakWidthModelOwner,PeakWidthModel>,
                doc_PeakWidthModelOwner_peakwidthmodel)
        ;
}

}   // namespace srrealmodule

// End of file
