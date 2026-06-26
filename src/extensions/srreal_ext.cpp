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
* srreal_ext - boost python interface to the srreal C++ codes in libdiffpy
*
*****************************************************************************/

#include <nanobind/nanobind.h>

#include "srreal_numpy_symbol.hpp"
#include <numpy/arrayobject.h>

// Declaration of the external wrappers --------------------------------------

namespace nb = nanobind;

namespace srrealmodule {

void wrap_libdiffpy_version(nb::module_& m);
void wrap_exceptions();
void wrap_EventTicker(nb::module_& m);
void wrap_Attributes(nb::module_& m);
void wrap_StructureDifference(nb::module_& m);
void wrap_StructureAdapter(nb::module_& m);
void wrap_AtomicStructureAdapter(nb::module_& m);
void wrap_ObjCrystAdapters(nb::module_& m);
void wrap_BaseBondGenerator(nb::module_& m);
void wrap_PairQuantity(nb::module_& m);
void wrap_PeakWidthModel(nb::module_& m);
void wrap_ScatteringFactorTable(nb::module_& m);
void wrap_PeakProfile(nb::module_& m);
void wrap_BVParametersTable(nb::module_& m);
void wrap_BVSCalculator(nb::module_& m);
void wrap_PDFBaseline(nb::module_& m);
void wrap_PDFEnvelope(nb::module_& m);
void wrap_PDFCalculators(nb::module_& m);
void wrap_BondCalculator(nb::module_& m);
void wrap_AtomRadiiTable(nb::module_& m);
void wrap_OverlapCalculator(nb::module_& m);

}   // namespace srrealmodule

namespace {

#if PY_MAJOR_VERSION >= 3
    void* initialize_numpy() { import_array(); return NULL; }
#else
    void initialize_numpy() { import_array(); }
#endif

}   // namespace

// Module Definitions --------------------------------------------------------

NB_MODULE(srreal_ext, m)
{
    using namespace srrealmodule;
    // initialize numpy module
    initialize_numpy();
    // execute external wrappers
    wrap_libdiffpy_version(m);
    wrap_exceptions();
    wrap_EventTicker(m);
    wrap_Attributes(m);
    wrap_StructureDifference(m);
    wrap_StructureAdapter(m);
    wrap_AtomicStructureAdapter(m);
    wrap_ObjCrystAdapters(m);
    wrap_BaseBondGenerator(m);
    wrap_PairQuantity(m);
    wrap_PeakWidthModel(m);
    wrap_ScatteringFactorTable(m);
    wrap_PeakProfile(m);
    wrap_BVParametersTable(m);
    wrap_BVSCalculator(m);
    wrap_PDFBaseline(m);
    wrap_PDFEnvelope(m);
    wrap_PDFCalculators(m);
    wrap_BondCalculator(m);
    wrap_AtomRadiiTable(m);
    wrap_OverlapCalculator(m);
    // load Python modules that tweak the wrapped classes
    nb::module_ srreal = nb::module_::import_("diffpy.srreal");
    if (!nb::hasattr(srreal, "_final_imports"))
    {
        PyObject* sysmods = PyImport_GetModuleDict();

        const char* fqname = "diffpy.srreal.srreal_ext";
        if (PyDict_GetItemString(sysmods, fqname) == nullptr) 
        {
            if (PyDict_SetItemString(sysmods, fqname, m.ptr()) < 0) 
            {
                nb::raise_python_error();
            }
        }

        nb::module_ final_imports =
            nb::module_::import_("diffpy.srreal._final_imports");

        nb::object import_now = final_imports.attr("import_now");
        import_now();
    }
}

// End of file
