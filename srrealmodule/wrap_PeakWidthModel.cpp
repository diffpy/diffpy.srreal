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

#include <string>
#include <boost/python.hpp>

#include <diffpy/srreal/PeakWidthModel.hpp>

#include "srreal_converters.hpp"

namespace srrealmodule {
namespace nswrap_PeakWidthModel {

using namespace boost;
using namespace boost::python;
using namespace diffpy::srreal;

// docstrings ----------------------------------------------------------------

const char* doc_PeakWidthModel_getRegisteredTypes = "\
Set of string identifiers for registered PeakWidthModel classes.\n\
These are allowed arguments for the setPeakWidthModel method.\n\
";

// wrappers ------------------------------------------------------------------

BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(getpwm_overloads,
        getPeakWidthModel, 0, 0)

DECLARE_PYSET_FUNCTION_WRAPPER(PeakWidthModel::getRegisteredTypes,
        getPeakWidthModelTypes_asset)

// Helper class allows overload of the PeakWidthModel methods from Python.

class PeakWidthModelWrap :
    public PeakWidthModel,
    public wrapper<PeakWidthModel>
{
    public:

        // HasClassRegistry methods

        PeakWidthModelPtr create() const
        {
            return this->get_override("create")();
        }

        PeakWidthModelPtr clone() const
        {
            return this->get_override("clone")();
        }

        const std::string& type() const
        {
            python::object tp = this->get_override("type")();
            mtype = python::extract<std::string>(tp);
            return mtype;
        }

        // own methods

        double calculate(const BaseBondGenerator& bnds) const
        {
            return this->get_override("calculate")(bnds);
        }

        double calculateFromMSD(double msdval) const
        {
            return this->get_override("calculateFromMSD")(msdval);
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
        noncopyable>("PeakWidthModel_ext")
        .def("create", &PeakWidthModel::create)
        .def("clone", &PeakWidthModel::clone)
        .def("type", pure_virtual(&PeakWidthModel::type),
                return_value_policy<copy_const_reference>())
        .def("calculate",
                pure_virtual(&PeakWidthModel::calculate))
        .def("calculateFromMSD",
                pure_virtual(&PeakWidthModel::calculateFromMSD))
        .def("_registerThisType", &PeakWidthModel::registerThisType)
        .def("createByType", &PeakWidthModel::createByType)
        .staticmethod("createByType")
        .def("getRegisteredTypes", getPeakWidthModelTypes_asset,
                doc_PeakWidthModel_getRegisteredTypes)
        .staticmethod("getRegisteredTypes")
        ;

    register_ptr_to_python<PeakWidthModelPtr>();

    class_<PeakWidthModelOwner>("PeakWidthModelOwner_ext")
        .def("getPeakWidthModel",
                (PeakWidthModelPtr(PeakWidthModelOwner::*)()) NULL,
                getpwm_overloads())
        .def("setPeakWidthModel",
                &PeakWidthModelOwner::setPeakWidthModel)
        .def("setPeakWidthModelByType",
                &PeakWidthModelOwner::setPeakWidthModelByType)
        ;
}

}   // namespace srrealmodule

// End of file
