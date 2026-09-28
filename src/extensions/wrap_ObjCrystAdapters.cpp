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
* Support pyobjcryst Crystal and Molecule objects through its Python API.
*
*****************************************************************************/

#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace srrealmodule {

void wrap_ObjCrystAdapters(nb::module_& m)
{
    // Preserve the extension's public entry points without sharing C++ objects
    // with pyobjcryst, which now embeds its own private ObjCryst++ library.
    m.def("convertObjCrystMolecule", [](nb::object molecule) {
        return nb::module_::import_("diffpy.srreal._objcrystconverters")
            .attr("convertObjCrystMolecule")(molecule);
    }, nb::arg("molecule"),
    "Copy a pyobjcryst Molecule to an AtomicStructureAdapter.");
    m.def("convertObjCrystCrystal", [](nb::object crystal) {
        return nb::module_::import_("diffpy.srreal._objcrystconverters")
            .attr("convertObjCrystCrystal")(crystal);
    }, nb::arg("crystal"),
    "Copy a pyobjcryst Crystal to a CrystalStructureAdapter.");
}

}  // namespace srrealmodule
