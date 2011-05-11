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
* $Id$
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

const char* doc_PeakWidthModel = "\
FIXME\n\
";

const char* doc_PeakWidthModel_create = "\
FIXME\n\
";

const char* doc_PeakWidthModel_clone = "\
FIXME\n\
";

const char* doc_PeakWidthModel_type = "\
FIXME\n\
";

const char* doc_PeakWidthModel_calculate = "\
FIXME\n\
";

const char* doc_PeakWidthModel_calculateFromMSD = "\
FIXME\n\
";

const char* doc_PeakWidthModel__registerThisType = "\
FIXME\n\
";

const char* doc_PeakWidthModel_createByType = "\
FIXME\n\
";

const char* doc_PeakWidthModel_getRegisteredTypes = "\
Set of string identifiers for registered PeakWidthModel classes.\n\
These are allowed arguments for the setPeakWidthModel method.\n\
";

const char* doc_PeakWidthModelOwner = "\
FIXME\n\
";

const char* doc_PeakWidthModelOwner_getPeakWidthModel = "\
FIXME\n\
";

const char* doc_PeakWidthModelOwner_setPeakWidthModel = "\
FIXME\n\
";

const char* doc_PeakWidthModelOwner_setPeakWidthModelByType = "\
FIXME\n\
";


// wrappers ------------------------------------------------------------------

BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(getpwm_overloads,
        getPeakWidthModel, 0, 0)

DECLARE_PYSET_FUNCTION_WRAPPER(PeakWidthModel::getRegisteredTypes,
        getPeakWidthModelTypes_asset)

// Helper class allows overload of the PeakWidthModel methods from Python.

class PeakWidthModelWrap :
    public PeakWidthModel,
    public wrapper_srreal<PeakWidthModel>
{
    public:

        // HasClassRegistry methods

        PeakWidthModelPtr create() const
        {
            return this->get_pure_virtual_override("create")();
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

        double calculateFromMSD(double msdval) const
        {
            override f = this->get_override("calculateFromMSD");
            if (f)  return f(msdval);
            return this->default_calculateFromMSD(msdval);
        }

        double default_calculateFromMSD(double msdval) const
        {
            return this->PeakWidthModel::calculateFromMSD(msdval);
        }

    private:

        mutable std::string mtype;

};  // class PeakWidthModelWrap

}   // namespace nswrap_PeakWidthModel

// Wrapper definition --------------------------------------------------------

void wrap_PeakWidthModel()
{
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
                &PeakWidthModel::calculate, doc_PeakWidthModel_calculate)
        .def("calculateFromMSD",
                &PeakWidthModel::calculateFromMSD,
                &PeakWidthModelWrap::default_calculateFromMSD,
                doc_PeakWidthModel_calculateFromMSD)
        .def("_registerThisType", &PeakWidthModel::registerThisType,
                doc_PeakWidthModel__registerThisType)
        .def("createByType", &PeakWidthModel::createByType,
                doc_PeakWidthModel_createByType)
        .staticmethod("createByType")
        .def("getRegisteredTypes", getPeakWidthModelTypes_asset,
                doc_PeakWidthModel_getRegisteredTypes)
        .staticmethod("getRegisteredTypes")
        ;

    register_ptr_to_python<PeakWidthModelPtr>();

    class_<PeakWidthModelOwner>("PeakWidthModelOwner", doc_PeakWidthModelOwner)
        .def("getPeakWidthModel",
                (PeakWidthModelPtr(PeakWidthModelOwner::*)()) NULL,
                getpwm_overloads(doc_PeakWidthModelOwner_getPeakWidthModel))
        .def("setPeakWidthModel",
                &PeakWidthModelOwner::setPeakWidthModel,
                doc_PeakWidthModelOwner_setPeakWidthModel)
        .def("setPeakWidthModelByType",
                &PeakWidthModelOwner::setPeakWidthModelByType,
                doc_PeakWidthModelOwner_setPeakWidthModelByType)
        ;
}

}   // namespace srrealmodule

// End of file
