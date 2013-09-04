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
* Bindings to the PeakWidthModel class.  The business methods can be overloaded
* from Python to create custom peak profiles.  This may be however quite slow.
*
*****************************************************************************/

#include <boost/python.hpp>
#include <string>

#include <diffpy/srreal/PeakWidthModel.hpp>

#include "srreal_converters.hpp"

namespace srrealmodule {
namespace nswrap_PeakWidthModel {

using namespace boost;
using namespace boost::python;
using namespace diffpy::srreal;

// docstrings ----------------------------------------------------------------

// FIXME merge in all unused ByType docstrings...

const char* doc_PeakWidthModel = "\
Base class for functors that calculate the PDF peak widths.\n\
Peak width is defined as full width at half maximum (FWHM).\n\
";

const char* doc_PeakWidthModel_create = "\
Return a new instance of the same PeakWidthModel type.\n\
This method must be overloaded in a derived class.\n\
";

const char* doc_PeakWidthModel_clone = "\
Return a duplicate of this PeakWidthModel instance.\n\
This method must be overloaded in a derived class.\n\
";

const char* doc_PeakWidthModel_type = "\
Return a unique string name for this PeakWidthModel class.\n\
This method must be overloaded in a derived class.\n\
";

const char* doc_PeakWidthModel_calculate = "\
Calculate the peak width for the specified bond.\n\
\n\
bnds -- instance of BaseBondGenerator with the current bond data.\n\
\n\
Return float.\n\
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

const char* doc_PeakWidthModel__registerThisType = "\
Add this instance to the global registry of PeakWidthModel types.\n\
\n\
No return value.  Cannot be overloaded in Python.\n\
";

const char* doc_PeakWidthModel_createByType = "\
Create a new PeakWidthModel instance of the specified type.\n\
\n\
tp   -- string identifier for a registered PeakWidthModel class.\n\
        Use getRegisteredTypes for a set of allowed values.\n\
\n\
Return new PeakWidthModel instance.\n\
";

const char* doc_PeakWidthModel_getRegisteredTypes = "\
Set of string identifiers for registered PeakWidthModel classes.\n\
These are the allowed arguments for the createByType method and\n\
setPeakWidthModelByType methods in the PDF calculator classes.\n\
";

const char* doc_PeakWidthModelOwner = "\
Base class for classes that own PeakWidthModel instance.\n\
";

const char* doc_PeakWidthModelOwner_peakwidthmodel = "\
PeakWidthModel object used for calculating the FWHM of the PDF peaks.\n\
This can be also set with the setPeakWidthModelByType method.\n\
";

const char* doc_PeakWidthModelOwner_setPeakWidthModelByType = "\
Set the peakwidthmodel attribute according to specified string type.\n\
\n\
tp   -- string identifier of a registered PeakWidthModel type.\n\
        Use PeakWidthModel.getRegisteredTypes for the allowed values.\n\
\n\
No return value.\n\
";


// wrappers ------------------------------------------------------------------

DECLARE_PYSET_FUNCTION_WRAPPER(PeakWidthModel::getRegisteredTypes,
        getPeakWidthModelTypes_asset)

// wrappers for the peakwidthmodel property

PeakWidthModelPtr getpwmodel(PeakWidthModelOwner& obj)
{
    return obj.getPeakWidthModel();
}

DECLARE_BYTYPE_SETTER_WRAPPER(setPeakWidthModel, setpwmodel)

// Helper class allows overload of the PeakWidthModel methods from Python.

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

        double maxWidth(StructureAdapterConstPtr stru,
                double rmin, double rmax) const
        {
            override f = this->get_pure_virtual_override("maxWidth");
            return f(stru, rmin, rmax);
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

}   // namespace nswrap_PeakWidthModel

// Wrapper definition --------------------------------------------------------

void wrap_PeakWidthModel()
{
    namespace bp = boost::python;
    using namespace nswrap_PeakWidthModel;
    using diffpy::Attributes;

    class_<PeakWidthModelWrap, bases<Attributes>,
        noncopyable>("PeakWidthModel", doc_PeakWidthModel)
        .def("create", &PeakWidthModel::create, doc_PeakWidthModel_create)
        .def("clone", &PeakWidthModel::clone, doc_PeakWidthModel_clone)
        .def("type", &PeakWidthModel::type,
                return_value_policy<copy_const_reference>(),
                doc_PeakWidthModel_type)
        .def("calculate",
                &PeakWidthModel::calculate,
                bp::arg("bnds"),
                doc_PeakWidthModel_calculate)
        .def("maxWidth",
                &PeakWidthModel::maxWidth,
                (bp::arg("stru"), bp::arg("rmin"), bp::arg("rmax")),
                doc_PeakWidthModel_maxWidth)
        .def("_registerThisType", &PeakWidthModel::registerThisType,
                doc_PeakWidthModel__registerThisType)
        .def("createByType", &PeakWidthModel::createByType,
                bp::arg("tp"), doc_PeakWidthModel_createByType)
        .staticmethod("createByType")
        .def("getRegisteredTypes", getPeakWidthModelTypes_asset,
                doc_PeakWidthModel_getRegisteredTypes)
        .staticmethod("getRegisteredTypes")
        ;

    register_ptr_to_python<PeakWidthModelPtr>();

    class_<PeakWidthModelOwner>("PeakWidthModelOwner", doc_PeakWidthModelOwner)
        .add_property("peakwidthmodel",
                getpwmodel,
                setpwmodel<PeakWidthModelOwner,PeakWidthModel>,
                doc_PeakWidthModelOwner_peakwidthmodel)
        ;
}

}   // namespace srrealmodule

// End of file
