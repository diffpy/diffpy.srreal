/*****************************************************************************
*
* diffpy.srreal     Complex Modeling Initiative
*                   Pavol Juhas
*                   (c) 2014 Brookhaven National Laboratory,
*                   Upton, New York.  All rights reserved.
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

#include <boost/python.hpp>
#include <boost/python/slice.hpp>

#include <diffpy/srreal/StructureAdapter.hpp>
#include <diffpy/srreal/StructureDifference.hpp>

#include "srreal_converters.hpp"

namespace srrealmodule {

// declarations
void sync_StructureDifference(boost::python::object obj);

namespace nswrap_StructureDifference {

using namespace boost;
using namespace diffpy::srreal;
using boost::python::slice;

// docstrings ----------------------------------------------------------------

const char* doc_StructureDifference = "\
FIXME\n\
";

const char* doc_StructureDifference_init_copy = "\
FIXME\n\
";

const char* doc_StructureDifference_init_structures = "\
FIXME\n\
";

const char* doc_StructureDifference_pop0 = "\
FIXME\n\
";

const char* doc_StructureDifference_add1 = "\
FIXME\n\
";

const char* doc_StructureDifference_diffmethod = "\
FIXME\n\
";

const char* doc_StructureDifference_allowsfastupdate = "\
FIXME\n\
";

// wrappers ------------------------------------------------------------------

python::list get_pop0(python::object obj)
{
    python::object pypop0 = obj.attr("_pop0");
    if (pypop0.ptr() == Py_None)
    {
        const StructureDifference& sd =
            python::extract<const StructureDifference&>(obj);
        pypop0 = obj.attr("_pop0") = convertToPythonList(sd.pop0);
    }
    return python::extract<python::list>(pypop0);
}


void set_pop0(python::object obj, python::object value)
{
    StructureDifference& sd = python::extract<StructureDifference&>(obj);
    sd.pop0 = extractintvector(value);
    get_pop0(obj)[slice()] = convertToPythonList(sd.pop0);
}


python::list get_add1(python::object obj)
{
    python::object pyadd1 = obj.attr("_add1");
    if (pyadd1.ptr() == Py_None)
    {
        const StructureDifference& sd =
            python::extract<const StructureDifference&>(obj);
        pyadd1 = obj.attr("_add1") = convertToPythonList(sd.add1);
    }
    return python::extract<python::list>(pyadd1);
}


void set_add1(python::object obj, python::object value)
{
    StructureDifference& sd = python::extract<StructureDifference&>(obj);
    sd.add1 = extractintvector(value);
    get_add1(obj)[slice()] = convertToPythonList(sd.add1);
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
    std::string emsg = "Unknown internal value of StructureDifference::Method.";
    throw std::out_of_range(emsg);
}


bool sd_allowsfastupdate(python::object obj)
{
    sync_StructureDifference(obj);
    StructureDifference& sd = python::extract<StructureDifference&>(obj);
    return sd.allowsfastupdate();
}

}   // namespace nswrap_StructureDifference

// this is a helper function to be called for Python-overridden
// StructureAdapter::diff method

void sync_StructureDifference(boost::python::object obj)
{
    using namespace boost::python;
    using diffpy::srreal::StructureDifference;
    StructureDifference& sd = extract<StructureDifference&>(obj);
    object pypop0 = obj.attr("_pop0");
    if (pypop0.ptr() != Py_None)  sd.pop0 = extractintvector(pypop0);
    object pyadd1 = obj.attr("_add1");
    if (pyadd1.ptr() != Py_None)  sd.add1 = extractintvector(pyadd1);
}

// Wrapper definitions -------------------------------------------------------

void wrap_StructureDifference()
{
    using namespace nswrap_StructureDifference;
    using namespace boost::python;
    namespace bp = boost::python;

    class_<StructureDifference>("StructureDifference", doc_StructureDifference)
        .def(init<const StructureDifference&>(bp::arg("sd"),
                    doc_StructureDifference_init_copy))
        .def(init<StructureAdapterPtr, StructureAdapterPtr>(
                    (bp::arg("adpt0"), bp::arg("adpt1")),
                    doc_StructureDifference_init_structures))
        .def_readwrite("stru0", &StructureDifference::stru0)
        .def_readwrite("stru1", &StructureDifference::stru1)
        .add_property("pop0",
                get_pop0, set_pop0, doc_StructureDifference_pop0)
        .setattr("_pop0", bp::object())
        .add_property("add1",
                get_add1, set_add1, doc_StructureDifference_add1)
        .setattr("_add1", bp::object())
        .add_property("diffmethod",
                get_diffmethod,
                doc_StructureDifference_diffmethod)
        .def("allowsfastupdate",
                sd_allowsfastupdate,
                doc_StructureDifference_allowsfastupdate)
        ;

}

}   // namespace srrealmodule

// End of file
