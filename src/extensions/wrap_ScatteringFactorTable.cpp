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
* Bindings to the ScatteringFactorTable class.  The business methods can be
* overridden from Python to return custom scattering factor values.
*
*****************************************************************************/

#include <nanobind/nanobind.h>
#include <nanobind/trampoline.h>

#include <diffpy/srreal/ScatteringFactorTable.hpp>
#include <diffpy/srreal/SFTXray.hpp>
#include <diffpy/srreal/SFTElectron.hpp>
#include <diffpy/srreal/SFTNeutron.hpp>
#include <diffpy/srreal/SFTElectronNumber.hpp>

#include "srreal_numpy_symbol.hpp"
// numpy/arrayobject.h needs to be included after srreal_numpy_symbol.hpp,
// which defines PY_ARRAY_UNIQUE_SYMBOL.  NO_IMPORT_ARRAY indicates
// import_array will be called in the extension module initializer.
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>


#include "srreal_converters.hpp"
#include "srreal_pickling.hpp"
#include "srreal_registry.hpp"

namespace nb = nanobind;

namespace srrealmodule {
namespace nswrap_ScatteringFactorTable {

using namespace diffpy::srreal;

// docstrings

const char* doc_ScatteringFactorTable = "\
Base class for looking up scattering factors by atom symbols.\n\
This class has virtual methods and cannot be used as is.\n\
\n\
A derived class has to override the following methods:\n\
\n\
    create(self)\n\
    clone(self)\n\
    type(self)\n\
    radiationType(self)\n\
    _standardLookup(self, smbl, q)\n\
\n\
Derived class can be added to the global registry of ScatteringFactorTable\n\
types by calling the _registerThisType method with any instance.\n\
";

const char* doc_ScatteringFactorTable_radiationType = "\
Return a string identifying the radiation type.\n\
'X' for x-rays, 'N' for neutrons.\n\
This method must be overridden in a derived class.\n\
";

const char* doc_ScatteringFactorTable_lookup = "\
Scattering factor of a specified atom at Q in 1/A.  The standard value\n\
can be redefined using the setCustomAs method.\n\
\n\
smbl -- string symbol for atom, ion or isotope.\n\
Q    -- Q value in inverse Angstroms, by default 0.\n\
        Q can be either float or NumPy array.\n\
\n\
Return float or NumPy array of the same shape as Q.\n\
No support for Python override.\n\
";

const char* doc_ScatteringFactorTable__standardLookup = "\
Standard value of the atom scattering factor at given Q in 1/A.\n\
\n\
smbl -- string symbol for atom, ion or isotope.\n\
q    -- scattering vector amplitude in 1/A.\n\
\n\
Return float.\n\
Raise ValueError for unknown atom symbol.\n\
This method must be overridden in a derived class.\n\
";

const char* doc_ScatteringFactorTable_setCustomAs2 = "\
Define custom alias for the specified atom symbol.\n\
Example: setCustomAs('12-C', 'C')  will declare the same\n\
scattering factors for '12-C' as for 'C'.\n\
\n\
smbl -- custom string alias for an existing standard symbol.\n\
src  -- standard atom symbol (cannot be another alias).\n\
\n\
No return value.  No support for Python override.\n\
";

const char* doc_ScatteringFactorTable_setCustomAs4 = "\
Define custom scattering factor for the specified atom symbol.\n\
The custom value is calculated by rescaling standard value\n\
from a source atom type.\n\
\n\
smbl -- string symbol of the atom with custom scattering factor.\n\
src  -- atom symbol for the source standard scattering factor.\n\
sf   -- new scattering factor value, defaults to the standard src factor.\n\
q    -- optional Q value for the new custom scattering factor.\n\
        The internal scaling of the standard value is calculated at this Q.\n\
\n\
No return value.  No support for Python override.\n\
";

const char* doc_ScatteringFactorTable_resetCustom = "\
Revert scattering factor for the specified symbol to a standard value.\n\
\n\
smbl -- string symbol for atom, ion or isotope.\n\
\n\
No return value.  No support for Python override.\n\
";

const char* doc_ScatteringFactorTable_resetAll = "\
Reset all custom scattering factor values.\n\
\n\
No return value.  No support for Python override.\n\
";

const char* doc_ScatteringFactorTable_getCustomSymbols = "\
Return a set of all atom symbols with custom scattering factors.\n\
";

const char* doc_ScatteringFactorTable_ticker = "\
Return EventTicker that marks last modification time of this object.\n\
This ticker object is used in fast PDF update, to check if scattering\n\
factors  changed since the last calculation.  The ticker.click() method\n\
thus needs to be used for _standardLookup that returns variable values.\n\
\n\
Return EventTicker object.\n\
This method can be overridden in a Python-derived class.\n\
";

const char* doc_ScatteringFactorTableOwner = "\
Base class for classes that own ScatteringFactorTable instance.\n\
";

const char* doc_ScatteringFactorTableOwner_scatteringfactortable = "\
ScatteringFactorTable object used for a lookup of scattering factors.\n\
This can be also set with the setScatteringFactorTableByType method.\n\
";

const char* doc_ScatteringFactorTableOwner_setScatteringFactorTableByType = "\
Set internal ScatteringFactorTable according to specified string type.\n\
\n\
tp   -- string identifier of a registered ScatteringFactorTable type.\n\
    Use ScatteringFactorTable.getRegisteredTypes for the allowed values.\n\
\n\
Deprecated: This method is deprecated and will be removed in the 2.0.0 release.\n\
Use direct assignment to the `scatteringfactortable` property instead, for example:\n\
    obj.scatteringfactortable = SFTNeutron()\n\
No return value.\n\
";

const char* doc_ScatteringFactorTableOwner_getRadiationType = "\
Return string identifying the radiation type.\n\
'X' for x-rays, 'N' for neutrons.\n\
";

const char* doc_SFTXray = "\
X-ray scattering factors table.\n\
\n\
Q-dependence of scattering factors is calculated using\n\
Waasmaier - Kirfel approximation valid up to Q=75/A.\n\
";

const char* doc_SFTElectron = "\
Electron scattering factors table.\n\
\n\
Q-dependence is derived from X-ray scattering factors according\n\
to the International Tables Volume C.\n\
";

const char* doc_SFTNeutron = "\
Table of neutron scattering lengths in fm.\n\
";

const char* doc_SFTElectronNumber = "\
Table of electron numbers for elements and ions.\n\
\n\
Can be used as Q-indendent scattering factors for X-rays.\n\
";

// wrappers ------------------------------------------------------------------

DECLARE_PYSET_METHOD_WRAPPER(getCustomSymbols, getCustomSymbols_asset)

// wrappers for the scatteringfactortable property

ScatteringFactorTablePtr getsftable(ScatteringFactorTableOwner& obj)
{
    return obj.getScatteringFactorTable();
}

DECLARE_BYTYPE_SETTER_WRAPPER(setScatteringFactorTable, setsftable)

// Helper class to support Python override of ScatteringFactorTable methods

class ScatteringFactorTableWrap :
    public ScatteringFactorTable
{
    public:

        NB_TRAMPOLINE(ScatteringFactorTable, 6);

        // Copy Constructor

        ScatteringFactorTableWrap() { }

        ScatteringFactorTableWrap(const ScatteringFactorTable& src)
        {
            ScatteringFactorTable& thistable = *this;
            // workaround for weird implicit ScatteringFactorTable::operator=
            // in g++ Red Hat 4.7.2-2, which must have non-constant argument.
            thistable = const_cast<ScatteringFactorTable&>(src);
        }

        // HasClassRegistry methods

        ScatteringFactorTablePtr create() const override
        {
            nb::gil_scoped_acquire gil;
            nb::detail::ticket ticket(nb_trampoline, "create", true);

            if (!ticket.key.is_valid()) 
            {
                throw nb::type_error(
                    "pure virtual method ScatteringFactorTable.create() called"
                );
            }

            nb::object rv = nb_trampoline.base().attr(ticket.key)();
            return mconfigurator.fetch(rv);
        }

        ScatteringFactorTablePtr clone() const override
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
                    "pure virtual method ScatteringFactorTable.type() called"
                );
            }

            nb::object tp = nb_trampoline.base().attr(ticket.key)();
            mtype = nb::cast<std::string>(tp);
            return mtype;
        }

        // own methods

        const std::string& radiationType() const override
        {
            nb::gil_scoped_acquire gil;
            nb::detail::ticket ticket(nb_trampoline, "radiationType", true);

            if (!ticket.key.is_valid()) 
            {
                throw nb::type_error(
                    "pure virtual method ScatteringFactorTable.radiationType() called"
                );
            }

            nb::object tp = nb_trampoline.base().attr(ticket.key)();
            mradiationtype = nb::cast<std::string>(tp);
            return mradiationtype;
        }

        double standardLookup(const std::string& smbl, double q) const override
        {
            NB_OVERRIDE_PURE_NAME("_standardLookup", standardLookup, smbl, q);
        }


        // Make the ticker method overridable from Python

        diffpy::eventticker::EventTicker& ticker() const
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
            return this->ScatteringFactorTable::ticker();
        }

    protected:

        // HasClassRegistry method

        void setupRegisteredObject(ScatteringFactorTablePtr p) const override
        {
            mconfigurator.setup(p);
        }

    private:

        mutable std::string mtype;
        mutable std::string mradiationtype;
        wrapper_registry_configurator<ScatteringFactorTable> mconfigurator;

        // serialization
        friend class boost::serialization::access;
        template<class Archive>
            void serialize(Archive& ar, const unsigned int version)
        {
            using boost::serialization::base_object;
            ar & base_object<ScatteringFactorTable>(*this);
        }

};  // class ScatteringFactorTableWrap

nb::object lookupnparray(const ScatteringFactorTable& sftb,
        std::string smbl, nb::object& qobj)
{
    NumPyArray_DoublePtr aa = extractNumPyDoubleArray(qobj);
    NumPyArray_DoublePtr bb = createNumPyDoubleArrayLike(aa.first);
    double* src = aa.second;
    double* last = aa.second + PyArray_Size(aa.first.ptr());
    double* dst = bb.second;
    for (; src != last; ++src, ++dst)
    {
        *dst = sftb.lookup(smbl, *src);
    }
    return bb.first;
}


}   // namespace nswrap_ScatteringFactorTable

// Wrapper definition --------------------------------------------------------
// TODO: rework pickle helpers

void wrap_ScatteringFactorTable(nb::module_& m)
{
    using namespace nswrap_ScatteringFactorTable;
    typedef ScatteringFactorTableOwner SFTOwner;

    nb::class_<ScatteringFactorTable, ScatteringFactorTableWrap>
        sftb(m, "ScatteringFactorTable", nb::dynamic_attr(), doc_ScatteringFactorTable);
    wrap_registry_methods(sftb)
        .def(nb::init<>())
        .def("radiationType",
                [](const ScatteringFactorTable &obj) 
                {
                    return std::string(obj.radiationType());
                },
                doc_ScatteringFactorTable_radiationType)
        .def("lookup",
                lookupnparray,
                nb::arg("smbl"), nb::arg("qarray"))
        .def("lookup",
                &ScatteringFactorTable::lookup,
                nb::arg("smbl"), nb::arg("q")=0.0,
                doc_ScatteringFactorTable_lookup)
        .def("_standardLookup",
                &ScatteringFactorTable::standardLookup,
                nb::arg("smbl"), nb::arg("q"),
                doc_ScatteringFactorTable__standardLookup)

        .def("setCustomAs", (void (ScatteringFactorTable::*)
                (const std::string&, const std::string&))
                &ScatteringFactorTable::setCustomAs,
                nb::arg("smbl"), nb::arg("src"),
                doc_ScatteringFactorTable_setCustomAs2)
        .def("setCustomAs", (void (ScatteringFactorTable::*)
                (const std::string&, const std::string&, double, double))
                &ScatteringFactorTable::setCustomAs,
                nb::arg("smbl"), nb::arg("src"),
                nb::arg("sf"), nb::arg("q")=0.0,
                doc_ScatteringFactorTable_setCustomAs4)

        .def("resetCustom", &ScatteringFactorTable::resetCustom,
                nb::arg("smbl"), doc_ScatteringFactorTable_resetCustom)
        .def("resetAll", &ScatteringFactorTable::resetAll,
                doc_ScatteringFactorTable_resetAll)
        .def("getCustomSymbols", getCustomSymbols_asset<ScatteringFactorTable>,
                doc_ScatteringFactorTable_getCustomSymbols)
        .def("ticker",
                &ScatteringFactorTable::ticker,
                nb::rv_policy::reference_internal,
                doc_ScatteringFactorTable_ticker)
        ;
        SerializationPickleSuite<ScatteringFactorTable, DICT_PICKLE>::bind(sftb);

    nb::class_<SFTXray, ScatteringFactorTable> sftxray(m,
            "SFTXray", doc_SFTXray);
    sftxray
        .def(nb::init<>())
        ;
        SerializationPickleSuite<SFTXray, DICT_IGNORE>::bind(sftxray);

    nb::class_<SFTElectron, ScatteringFactorTable> sftelectron(m,
            "SFTElectron", doc_SFTElectron);
    sftelectron
        .def(nb::init<>())
        ;
        SerializationPickleSuite<SFTElectron, DICT_IGNORE>::bind(sftelectron);

    nb::class_<SFTNeutron, ScatteringFactorTable> sftneutron(m,
            "SFTNeutron", doc_SFTNeutron);
    sftneutron
        .def(nb::init<>())
        ;
        SerializationPickleSuite<SFTNeutron, DICT_IGNORE>::bind(sftneutron);

    nb::class_<SFTElectronNumber, ScatteringFactorTable> sftelectronnumber(m,
            "SFTElectronNumber", doc_SFTElectronNumber);
    sftelectronnumber
        .def(nb::init<>())
        ;
        SerializationPickleSuite<SFTElectronNumber, DICT_IGNORE>::bind(sftelectronnumber);

    nb::class_<ScatteringFactorTableOwner>(m, "ScatteringFactorTableOwner",
            doc_ScatteringFactorTableOwner)
        .def(nb::init<>())
        .def_prop_rw("scatteringfactortable",
                getsftable,
                setsftable<ScatteringFactorTableOwner,ScatteringFactorTable>,
                doc_ScatteringFactorTableOwner_scatteringfactortable)
        // deprecated: prefer assigning the `scatteringfactortable` property
        .def("setScatteringFactorTableByType",
            +[](SFTOwner& obj, const std::string& tp)
            {
                try
                {
                    nb::object warnings = nb::module_::import_("warnings");
                    nb::object builtins = nb::module_::import_("builtins");
                    nb::object DeprecationWarning = builtins.attr("DeprecationWarning");
                    warnings.attr("warn")(
                        std::string("setScatteringFactorTableByType is deprecated; "
                                "assign the 'scatteringfactortable' property directly, for example:\n"
                                "obj.scatteringfactortable = SFTNeutron()"),
                        DeprecationWarning,
                        2);
                }
                catch (...) { /* don't let warnings break the binding */ }
                obj.setScatteringFactorTableByType(tp);
            },
            nb::arg("tp"),
            doc_ScatteringFactorTableOwner_setScatteringFactorTableByType)
        .def("getRadiationType",
                [](const SFTOwner &obj) 
                {
                    return std::string(obj.getRadiationType());
                },
                doc_ScatteringFactorTableOwner_getRadiationType)
        ;
}

}   // namespace srrealmodule

// Serialization -------------------------------------------------------------

BOOST_CLASS_EXPORT(srrealmodule::nswrap_ScatteringFactorTable::ScatteringFactorTableWrap)

// End of file
