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
* Bindings to the ScatteringFactorTable class.  The business methods can be
* overloaded from Python to create custom peak profiles.
*
* $Id$
*
*****************************************************************************/

#include <boost/python.hpp>

#include <diffpy/srreal/ScatteringFactorTable.hpp>
#include "srreal_converters.hpp"

namespace srrealmodule {
namespace nswrap_ScatteringFactorTable {

using namespace boost;
using namespace boost::python;
using namespace diffpy::srreal;

// docstrings

const char* doc_ScatteringFactorTable_getRegisteredTypes = "\
Set of string identifiers for registered ScatteringFactorTable classes.\n\
These are allowed arguments for the createByType method and\n\
setScatteringFactorTableByType methods in PDF calculator classes.\n\
";

// wrappers

BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(getsft_overloads,
        getScatteringFactorTable, 0, 0)

DECLARE_PYSET_FUNCTION_WRAPPER(ScatteringFactorTable::getRegisteredTypes,
        getScatteringFactorTableTypes_asset)

// Helper class for overloads of ScatteringFactorTable methods from Python

class ScatteringFactorTableWrap :
    public ScatteringFactorTable,
    public wrapper<ScatteringFactorTable>
{
    public:

        // HasClassRegistry methods

        ScatteringFactorTablePtr create() const
        {
            return this->get_override("create")();
        }

        ScatteringFactorTablePtr clone() const
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

        const std::string& radiationType() const
        {
            python::object tp = this->get_override("radiationType")();
            mradiationtype = python::extract<std::string>(tp);
            return mradiationtype;
        }


        double fetch(const std::string& smbl) const
        {
            return this->get_override("_fetch")(smbl);
        }

    private:

        mutable std::string mtype;
        mutable std::string mradiationtype;

};  // class ScatteringFactorTableWrap

}   // namespace nswrap_ScatteringFactorTable

// Wrapper definition --------------------------------------------------------

void wrap_ScatteringFactorTable()
{
    using namespace nswrap_ScatteringFactorTable;
    typedef ScatteringFactorTableOwner SFTOwner;

    // NOTE: fetch is protected and so the _fetch wrapper raises exception for
    // classes defined in C++.  However, _fetch can be overloaded for classes
    // derived in Python.

    class_<ScatteringFactorTableWrap, noncopyable>("ScatteringFactorTable_ext")
        .def("create", &ScatteringFactorTable::create)
        .def("clone", &ScatteringFactorTable::clone)
        .def("type", pure_virtual(&ScatteringFactorTable::type),
                return_value_policy<copy_const_reference>())
        .def("radiationType",
                pure_virtual(&ScatteringFactorTable::radiationType),
                return_value_policy<copy_const_reference>())
        .def("lookup", &ScatteringFactorTable::lookup,
                return_value_policy<copy_const_reference>())
        .def("setCustom", &ScatteringFactorTable::setCustom)
        .def("resetCustom", &ScatteringFactorTable::resetCustom)
        .def("resetAll", &ScatteringFactorTable::resetAll)
        .def("_fetch", pure_virtual(&ScatteringFactorTableWrap::fetch))
        .def("_registerThisType", &ScatteringFactorTable::registerThisType)
        .def("createByType", &ScatteringFactorTable::createByType)
        .staticmethod("createByType")
        .def("getRegisteredTypes", getScatteringFactorTableTypes_asset,
                doc_ScatteringFactorTable_getRegisteredTypes)
        .staticmethod("getRegisteredTypes")
        ;

    register_ptr_to_python<ScatteringFactorTablePtr>();

    class_<ScatteringFactorTableOwner>("ScatteringFactorTableOwner_ext")
        .def("getScatteringFactorTable",
                (ScatteringFactorTablePtr(SFTOwner::*)()) NULL,
                getsft_overloads())
        .def("setScatteringFactorTable",
                &SFTOwner::setScatteringFactorTable)
        .def("setScatteringFactorTableByType",
                &SFTOwner::setScatteringFactorTableByType)
        .def("getRadiationType",
                &SFTOwner::getRadiationType,
                return_value_policy<copy_const_reference>())
        ;
}

}   // namespace srrealmodule

// End of file
