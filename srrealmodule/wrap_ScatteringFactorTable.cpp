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
Base class for looking up scattering factors of atoms,\n\
ions and isotopes.\n\
";

const char* doc_ScatteringFactorTable___init__ = "\
Initialize a new ScatteringFactorTable.  This method needs to\n\
be called from a derived class.  This class has virtual\n\
methods and cannot be used as is.\n\
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
Return a string identifying the radiation type, 'X' for x-rays,\n\
'N' for neutrons.  This method must be overloaded in a derived class.\n\
";

const char* doc_ScatteringFactorTable_lookup = "\
Scattering factor of a specified atom at Q=0/A.  The standard value\n\
can be overloaded using the setCustom method.  Otherwise the same as\n\
lookupatq(smbl, 0)\n\
\n\
smbl -- string symbol for atom, ion or isotope\n\
\n\
Return float.\n\
Note: used by PDFCalculator class.\n\
";

const char* doc_ScatteringFactorTable_lookupatq = "\
Scattering factor of a specified atom at given Q in 1/A.\n\
\n\
smbl -- string symbol for atom, ion or isotope\n\
q    -- scattering vector amplitude in 1/A\n\
\n\
Return float.\n\
This method must be overloaded in a derived class.\n\
Note: used by DebyePDFCalculator class.\n\
";

const char* doc_ScatteringFactorTable_setCustom = "\
Define custom scattering factor for the specified symbol.\n\
\n\
smbl -- string symbol for atom, ion or isotope\n\
sf   -- new scattering factor value\n\
\n\
No return value.\n\
";

const char* doc_ScatteringFactorTable_resetCustom = "\
Revert scattering factor for the specified symbol to a standard value.\n\
\n\
smbl -- string symbol for atom, ion or isotope\n\
\n\
No return value.\n\
";

const char* doc_ScatteringFactorTable_resetAll = "\
Reset all custom scattering factor values.\n\
";

const char* doc_ScatteringFactorTable__registerThisType = "\
Add this instance to a global registry of ScatteringFactorTable types.\n\
";

const char* doc_ScatteringFactorTable_createByType = "\
Create a new ScatteringFactorTable instance of the specified type.\n\
\n\
tp   -- string identifier for a registered ScatteringFactorTable\n\
        Use getRegisteredTypes for a set of allowed values.\n\
\n\
Return a ScatteringFactorTable instance\n\
";


const char* doc_ScatteringFactorTable_getRegisteredTypes = "\
Return a set of string names for the registered ScatteringFactorTable\n\
types.  These are allowed arguments for the createByType method and\n\
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
        .def(init<const ScatteringFactorTable&>(bp::arg("self"),
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
                doc_ScatteringFactorTable_lookup)
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
