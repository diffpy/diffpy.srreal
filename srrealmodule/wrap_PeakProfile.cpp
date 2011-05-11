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
* Bindings to the PeakProfile class.  The business methods can be overloaded
* from Python to create custom peak profiles.  This may be however quite slow.
*
* $Id$
*
*****************************************************************************/

#include <boost/python.hpp>
#include <string>

#include <diffpy/srreal/PeakProfile.hpp>

#include "srreal_converters.hpp"

namespace srrealmodule {
namespace nswrap_PeakProfile {

using namespace boost;
using namespace boost::python;
using namespace diffpy::srreal;

// docstrings ----------------------------------------------------------------

const char* doc_PeakProfile = "\
FIXME\n\
";

const char* doc_PeakProfile_create = "\
FIXME\n\
";

const char* doc_PeakProfile_clone = "\
FIXME\n\
";

const char* doc_PeakProfile_type = "\
FIXME\n\
";

const char* doc_PeakProfile_yvalue = "\
FIXME\n\
";

const char* doc_PeakProfile_xboundlo = "\
FIXME\n\
";

const char* doc_PeakProfile_xboundhi = "\
FIXME\n\
";

const char* doc_PeakProfile__registerThisType = "\
FIXME\n\
";

const char* doc_PeakProfile_createByType = "\
FIXME\n\
";

const char* doc_PeakProfile_getRegisteredTypes = "\
Set of string identifiers for registered PeakProfile classes.\n\
These are allowed arguments for the createByType static method.\n\
";

// wrappers ------------------------------------------------------------------

DECLARE_PYSET_FUNCTION_WRAPPER(PeakProfile::getRegisteredTypes,
        getPeakProfileTypes_asset)

// Helper class allows overload of the PeakProfile methods from Python.

class PeakProfileWrap :
    public PeakProfile,
    public wrapper_srreal<PeakProfile>
{
    public:

        // HasClassRegistry methods

        PeakProfilePtr create() const
        {
            return this->get_pure_virtual_override("create")();
        }

        PeakProfilePtr clone() const
        {
            return this->get_pure_virtual_override("clone")();
        }


        const std::string& type() const
        {
            python::object tp = this->get_pure_virtual_override("type")();
            mtype = python::extract<std::string>(tp);
            return mtype;
        }

        // own methods

        double yvalue(double x, double fwhm) const
        {
            return this->get_pure_virtual_override("yvalue")(x, fwhm);
        }

        double xboundlo(double fwhm) const
        {
            return this->get_pure_virtual_override("xboundlo")(fwhm);
        }

        double xboundhi(double fwhm) const
        {
            return this->get_pure_virtual_override("xboundhi")(fwhm);
        }

    private:

        mutable std::string mtype;

};  // class PeakProfileWrap

}   // namespace nswrap_PeakProfile

// Wrapper definition --------------------------------------------------------

void wrap_PeakProfile()
{
    using namespace nswrap_PeakProfile;
    using diffpy::Attributes;

    class_<PeakProfileWrap, bases<Attributes>,
        noncopyable>("PeakProfile", doc_PeakProfile)
        .def("create", &PeakProfile::create, doc_PeakProfile_create)
        .def("clone", &PeakProfile::clone, doc_PeakProfile_clone)
        .def("type", &PeakProfile::type,
                return_value_policy<copy_const_reference>(),
                doc_PeakProfile_type)
        .def("yvalue", &PeakProfile::yvalue, doc_PeakProfile_yvalue)
        .def("xboundlo", &PeakProfile::xboundlo, doc_PeakProfile_xboundlo)
        .def("xboundhi", &PeakProfile::xboundhi, doc_PeakProfile_xboundhi)
        .def("_registerThisType", &PeakProfile::registerThisType,
                doc_PeakProfile__registerThisType)
        .def("createByType", &PeakProfile::createByType,
                doc_PeakProfile_createByType)
        .staticmethod("createByType")
        .def("getRegisteredTypes", getPeakProfileTypes_asset,
                doc_PeakProfile_getRegisteredTypes)
        .staticmethod("getRegisteredTypes")
        ;

    register_ptr_to_python<PeakProfilePtr>();
}

}   // namespace srrealmodule

// End of file
