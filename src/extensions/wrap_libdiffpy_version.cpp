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
* Access to the libdiffpy version data.
*
*****************************************************************************/

#include <diffpy/version.hpp>
#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace srrealmodule {
namespace nswrap_libdiffpy_version {

// docstrings ----------------------------------------------------------------

const char* doc__get_libdiffpy_version_info_dict = "\
Return dictionary with version data for the loaded libdiffpy library.\n\
";

// wrappers ------------------------------------------------------------------

nb::dict get_libdiffpy_version_info_dict()
{
    nb::dict rv;
    // Obtain version data from runtime values.
    rv["version"] = libdiffpy_version_info::version;
    rv["version_str"] = libdiffpy_version_info::version_str;
    rv["major"] = libdiffpy_version_info::major;
    rv["minor"] = libdiffpy_version_info::minor;
    rv["micro"] = libdiffpy_version_info::micro;
    rv["date"] = libdiffpy_version_info::date;
    rv["git_commit"] = libdiffpy_version_info::git_sha;
    rv["patch"] = libdiffpy_version_info::patch;
    return rv;
}

}   // namespace nswrap_libdiffpy_version

// Wrapper definition --------------------------------------------------------

void wrap_libdiffpy_version(nb::module_& m)
{
    using namespace nswrap_libdiffpy_version;

    m.def("_get_libdiffpy_version_info_dict",
            get_libdiffpy_version_info_dict,
            doc__get_libdiffpy_version_info_dict);

}

}   // namespace srrealmodule

// End of file
