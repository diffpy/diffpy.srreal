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

const char* doc_ScatteringFactorTable = "\
Base class for looking up scattering factors by atom symbols.\n\
This class has virtual methods and cannot be used as is.\n\
\n\
A derived class has to overload the following methods:\n\
\n\
    create(self)\n\
    clone(self)\n\
    type(self)\n\
    radiationType(self)\n\
    lookupatq(self, smbl, q)\n\
\n\
Derived class can be added to the global registry of ScatteringFactorTable\n\
types by calling the _registerThisType method with any instance.\n\
";

const char* doc_ScatteringFactorTable___init__ = "\
Initialize ScatteringFactorTable and create the internal C++ object.\n\
\n\
src  -- copy custom scattering factors from this instance if specified.\n\
        src is necessary for proper overloading of the clone method.\n\
\n\
No return value.\n\
";

const char* doc_ScatteringFactorTable_create = "\
Return a new instance of the same ScatteringFactorTable type.\n\
This method must be overloaded in a derived class.\n\
";

const char* doc_ScatteringFactorTable_clone = "\
Return a duplicate of this ScatteringFactorTable instance.\n\
This method must be overloaded in a derived class.\n\
";

const char* doc_ScatteringFactorTable_type = "\
Return a unique string name for this ScatteringFactorTable class.\n\
This method must be overloaded in a derived class.\n\
";

const char* doc_ScatteringFactorTable_radiationType = "\
Return a string identifying the radiation type.\n\
'X' for x-rays, 'N' for neutrons.\n\
This method must be overloaded in a derived class.\n\
";

const char* doc_ScatteringFactorTable_lookup = "\
Scattering factor of a specified atom at Q=0/A.  The standard value\n\
can be overloaded using the setCustom method.  Otherwise the same as\n\
lookupatq(smbl, 0)\n\
\n\
smbl -- string symbol for atom, ion or isotope\n\
\n\
Return float.  Cannot be overloaded in Python.\n\
Note: Used by PDFCalculator.\n\
";

const char* doc_ScatteringFactorTable_lookupatq = "\
Scattering factor of a specified atom at given Q in 1/A.\n\
\n\
smbl -- string symbol for atom, ion or isotope\n\
q    -- scattering vector amplitude in 1/A\n\
\n\
Return float.\n\
This method must be overloaded in a derived class.\n\
Note: Used by DebyePDFCalculator.\n\
";

const char* doc_ScatteringFactorTable_setCustom = "\
Define custom scattering factor for the specified symbol.\n\
\n\
smbl -- string symbol for atom, ion or isotope\n\
sf   -- new scattering factor value\n\
\n\
No return value.  Cannot be overloaded in Python.\n\
";

const char* doc_ScatteringFactorTable_resetCustom = "\
Revert scattering factor for the specified symbol to a standard value.\n\
\n\
smbl -- string symbol for atom, ion or isotope\n\
\n\
No return value.  Cannot be overloaded in Python.\n\
";

const char* doc_ScatteringFactorTable_resetAll = "\
Reset all custom scattering factor values.\n\
\n\
No return value.  Cannot be overloaded in Python.\n\
";

const char* doc_ScatteringFactorTable__registerThisType = "\
Add this instance to the global registry of ScatteringFactorTable types.\n\
\n\
No return value.  Cannot be overloaded in Python.\n\
";

const char* doc_ScatteringFactorTable_createByType = "\
Create a new ScatteringFactorTable instance of the specified type.\n\
\n\
tp   -- string identifier for a registered ScatteringFactorTable\n\
        Use getRegisteredTypes for a set of allowed values.\n\
\n\
Return new ScatteringFactorTable instance.\n\
";

const char* doc_ScatteringFactorTable_getRegisteredTypes = "\
Return a set of string names for the registered ScatteringFactorTable\n\
types.  These are allowed arguments for the createByType method and\n\
setScatteringFactorTableByType methods in PDF calculator classes.\n\
";

const char* doc_ScatteringFactorTableOwner = "\
Base class for classes that own ScatteringFactorTable instance.\n\
";

const char* doc_ScatteringFactorTableOwner_getScatteringFactorTable = "\
Return the internal ScatteringFactorTable.\n\
";

const char* doc_ScatteringFactorTableOwner_setScatteringFactorTable = "\
Set internal ScatteringFactorTable to the specified instance.\n\
\n\
sftable  -- an instance of ScatteringFactorTable\n\
\n\
No return value.\n\
";

const char* doc_ScatteringFactorTableOwner_setScatteringFactorTableByType = "\
Set internal ScatteringFactorTable according to specified string type.\n\
\n\
tp   -- string identifier of a registered ScatteringFactorTable type\n\
        Use ScatteringFactorTable.getRegisteredTypes for allowed values.\n\
\n\
No return value.\n\
";

const char* doc_ScatteringFactorTableOwner_getRadiationType = "\
Return string identifying the radiation type.\n\
'X' for x-rays, 'N' for neutrons.\n\
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

        // Copy Constructor

        ScatteringFactorTableWrap() { }

        ScatteringFactorTableWrap(const ScatteringFactorTable& src)
        {
            ScatteringFactorTable& thistable = *this;
            thistable = src;
        }

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


        double lookupatq(const std::string& smbl, double q) const
        {
            return this->get_override("lookupatq")(smbl, q);
        }

    private:

        mutable std::string mtype;
        mutable std::string mradiationtype;

};  // class ScatteringFactorTableWrap

}   // namespace nswrap_ScatteringFactorTable

// Wrapper definition --------------------------------------------------------

void wrap_ScatteringFactorTable()
{
    namespace bp = boost::python;
    using namespace nswrap_ScatteringFactorTable;
    typedef ScatteringFactorTableOwner SFTOwner;

    class_<ScatteringFactorTableWrap, noncopyable>(
            "ScatteringFactorTable", doc_ScatteringFactorTable)
        .def(init<const ScatteringFactorTable&>(bp::arg("src"),
                    doc_ScatteringFactorTable___init__))
        .def("create", pure_virtual(&ScatteringFactorTable::create),
                doc_ScatteringFactorTable_create)
        .def("clone", pure_virtual(&ScatteringFactorTable::clone),
                doc_ScatteringFactorTable_clone)
        .def("type", pure_virtual(&ScatteringFactorTable::type),
                return_value_policy<copy_const_reference>(),
                doc_ScatteringFactorTable_type)
        .def("radiationType",
                pure_virtual(&ScatteringFactorTable::radiationType),
                return_value_policy<copy_const_reference>(),
                doc_ScatteringFactorTable_radiationType)
        .def("lookup",
                &ScatteringFactorTable::lookup,
                bp::arg("smbl"), doc_ScatteringFactorTable_lookup)
        .def("lookupatq",
                pure_virtual(&ScatteringFactorTable::lookupatq),
                (bp::arg("smbl"), bp::arg("q")),
                doc_ScatteringFactorTable_lookupatq)
        .def("setCustom", &ScatteringFactorTable::setCustom,
                (bp::arg("smbl"), bp::arg("sf")),
                doc_ScatteringFactorTable_setCustom)
        .def("resetCustom", &ScatteringFactorTable::resetCustom,
                bp::arg("smbl"), doc_ScatteringFactorTable_setCustom)
        .def("resetAll", &ScatteringFactorTable::resetAll,
                doc_ScatteringFactorTable_resetAll)
        .def("_registerThisType", &ScatteringFactorTable::registerThisType,
                doc_ScatteringFactorTable__registerThisType)
        .def("createByType", &ScatteringFactorTable::createByType,
                bp::arg("tp"), doc_ScatteringFactorTable_createByType)
        .staticmethod("createByType")
        .def("getRegisteredTypes", getScatteringFactorTableTypes_asset,
                doc_ScatteringFactorTable_getRegisteredTypes)
        .staticmethod("getRegisteredTypes")
        ;

    register_ptr_to_python<ScatteringFactorTablePtr>();

    class_<ScatteringFactorTableOwner>("ScatteringFactorTableOwner",
            doc_ScatteringFactorTableOwner)
        .def("getScatteringFactorTable",
                (ScatteringFactorTablePtr(SFTOwner::*)()) NULL,
                getsft_overloads(
                    doc_ScatteringFactorTableOwner_getScatteringFactorTable))
        .def("setScatteringFactorTable",
                &SFTOwner::setScatteringFactorTable,
                bp::arg("sftable"),
                doc_ScatteringFactorTableOwner_setScatteringFactorTable)
        .def("setScatteringFactorTableByType",
                &SFTOwner::setScatteringFactorTableByType,
                bp::arg("tp"),
                doc_ScatteringFactorTableOwner_setScatteringFactorTableByType)
        .def("getRadiationType",
                &SFTOwner::getRadiationType,
                return_value_policy<copy_const_reference>(),
                doc_ScatteringFactorTableOwner_getRadiationType)
        ;
}

}   // namespace srrealmodule

// End of file
