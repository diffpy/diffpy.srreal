/*****************************************************************************
*
* diffpy.srreal     Complex Modeling Initiative
*                   (c) 2014 Brookhaven Science Associates,
*                   Brookhaven National Laboratory.
*                   All rights reserved.
*
* File coded by:    Pavol Juhas
*
* See AUTHORS.txt for a list of people who contributed.
* See LICENSE.txt for license information.
*
******************************************************************************
*
* Wrap StructureDifference class for expressing differences between two
* StructureAdapter objects.
*
*****************************************************************************/

#include <nanobind/nanobind.h>

#include <cstdlib>

#include <diffpy/srreal/StructureAdapter.hpp>
#include <diffpy/srreal/StructureDifference.hpp>

#include "srreal_converters.hpp"

namespace nb = nanobind;

namespace srrealmodule {

// declarations
void sync_StructureDifference(nb::object obj);

namespace nswrap_StructureDifference {

using namespace diffpy::srreal;

// docstrings ----------------------------------------------------------------

const char* doc_StructureDifference = "\
Class for expressing difference between two StructureAdapter objects.\n\
\n\
Attributes:\n\
\n\
stru0    -- old StructureAdapter instance\n\
stru1    -- new StructureAdapter instance\n\
pop0     -- list of indices of atoms that are only in stru0\n\
add1     -- indices of atoms that are only in stru1\n\
";

const char* doc_StructureDifference_init_copy = "\
Create a copy of an existing StructureDifference object sd.\n\
";

const char* doc_StructureDifference_init_structures = "\
Create StructureDifference for comparing stru0 (old) and stru1 (new).\n\
";

const char* doc_StructureDifference_diffmethod = "\
Read-only string type of C++ difference algorithm that was used to compare\n\
the structures.  Possible values are ('NONE', 'SIDEBYSIDE', 'SORTED').\n\
";

const char* doc_StructureDifference_allowsfastupdate = "\
Return True if PairQuantity evaluated for stru0 can be fast-updated\n\
for structure stru1.  Fast update is done by removal of contributions\n\
from stru0 atoms at indices pop0 and addition of add1 atoms in stru1.\n\
";

// wrappers ------------------------------------------------------------------

nb::list get_pop0(nb::object obj)
{
    nb::object pypop0 = obj.attr("_pop0");
    if (pypop0.is_none())
    {
        const StructureDifference& sd =
            nb::cast<const StructureDifference&>(obj);
        pypop0 = obj.attr("_pop0") = convertToPythonList(sd.pop0);
    }
    return nb::borrow<nb::list>(pypop0);
}


void set_pop0(nb::object obj, nb::object value)
{
    StructureDifference& sd = nb::cast<StructureDifference&>(obj);
    sd.pop0 = extractintvector(value);
    nb::cast<nb::object>(get_pop0(obj))[nb::slice(nb::none(), nb::none(), nb::none())] = convertToPythonList(sd.pop0);
}


nb::list get_add1(nb::object obj)
{
    nb::object pyadd1 = obj.attr("_add1");
    if (pyadd1.is_none())
    {
        const StructureDifference& sd =
            nb::cast<const StructureDifference&>(obj);
        pyadd1 = obj.attr("_add1") = convertToPythonList(sd.add1);
    }
    return nb::borrow<nb::list>(pyadd1);
}


void set_add1(nb::object obj, nb::object value)
{
    StructureDifference& sd = nb::cast<StructureDifference&>(obj);
    sd.add1 = extractintvector(value);
    nb::cast<nb::object>(get_add1(obj))[nb::slice(nb::none(), nb::none(), nb::none())] = convertToPythonList(sd.add1);
}


std::string get_diffmethod(const StructureDifference& sd)
{
    switch (sd.diffmethod)
    {
        case StructureDifference::Method::NONE:
            return "NONE";
        case StructureDifference::Method::SIDEBYSIDE:
            return "SIDEBYSIDE";
        case StructureDifference::Method::SORTED:
            return "SORTED";
    }
    const char* emsg = "Unknown internal value of StructureDifference::Method.";
    PyErr_SetString(PyExc_NotImplementedError, emsg);
    throw nb::python_error();
}


bool sd_allowsfastupdate(nb::object obj)
{
    sync_StructureDifference(obj);
    StructureDifference& sd = nb::cast<StructureDifference&>(obj);
    return sd.allowsfastupdate();
}

}   // namespace nswrap_StructureDifference

// this is a helper function to be called for Python-overridden
// StructureAdapter::diff method

void sync_StructureDifference(nb::object obj)
{
    using diffpy::srreal::StructureDifference;
    StructureDifference& sd = nb::cast<StructureDifference&>(obj);
    nb::object pypop0 = obj.attr("_pop0");
    if (!pypop0.is_none())  sd.pop0 = extractintvector(pypop0);
    nb::object pyadd1 = obj.attr("_add1");
    if (!pyadd1.is_none())  sd.add1 = extractintvector(pyadd1);
}

// Wrapper definitions -------------------------------------------------------

void wrap_StructureDifference(nb::module_& m)
{
    using namespace nswrap_StructureDifference;

    nb::class_<StructureDifference> sd(m, "StructureDifference",
            doc_StructureDifference, nb::dynamic_attr());
    sd
        .def(nb::init<>())
        .def(nb::init<const StructureDifference&>(), nb::arg("sd"),
                    doc_StructureDifference_init_copy)
        .def(nb::init<StructureAdapterPtr, StructureAdapterPtr>(), 
                    nb::arg("stru0"), nb::arg("stru1"),
                    doc_StructureDifference_init_structures)
        .def_rw("stru0", &StructureDifference::stru0)
        .def_rw("stru1", &StructureDifference::stru1)
        .def_prop_rw("pop0", get_pop0, set_pop0)
        .def_prop_rw("add1", get_add1, set_add1)
        .def_prop_ro("diffmethod",
                get_diffmethod,
                doc_StructureDifference_diffmethod)
        .def("allowsfastupdate",
                sd_allowsfastupdate,
                doc_StructureDifference_allowsfastupdate)
        ;

    sd.attr("_pop0") = nb::none();
    sd.attr("_add1") = nb::none();
}

}   // namespace srrealmodule

// End of file
