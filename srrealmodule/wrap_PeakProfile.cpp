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

#include <string>
#include <boost/python.hpp>

#include <diffpy/srreal/PeakProfile.hpp>

#include "srreal_converters.hpp"

namespace srrealmodule {
namespace nswrap_PeakProfile {

using namespace boost;
using namespace boost::python;
using namespace diffpy::srreal;

// Helper class allows overload of the PeakProfile methods from Python.

class PeakProfileWrap :
    public PeakProfile,
    public wrapper<PeakProfile>
{
    public:

        // constructors

        PeakProfilePtr create() const
        {
            return this->get_override("_create")();
        }

        PeakProfilePtr clone() const
        {
            return this->get_override("_clone")();
        }

        // methods

        const std::string& type() const
        {
            python::object tp = this->get_override("type")();
            mtype = python::extract<std::string>(tp);
            return mtype;
        }

        double yvalue(double x, double fwhm) const
        {
            return this->get_override("yvalue")(x, fwhm);
        }

        double xboundlo(double fwhm) const
        {
            return this->get_override("xboundlo")(fwhm);
        }

        double xboundhi(double fwhm) const
        {
            return this->get_override("xboundhi")(fwhm);
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
        noncopyable>("PeakProfile_ext")
        .def("_create", &PeakProfile::create)
        .def("_clone", &PeakProfile::clone)
        .def("type", pure_virtual(&PeakProfile::type),
                return_value_policy<copy_const_reference>())
        .def("yvalue", pure_virtual(&PeakProfile::yvalue))
        .def("xboundlo", pure_virtual(&PeakProfile::xboundlo))
        .def("xboundhi", pure_virtual(&PeakProfile::xboundhi))
        ;

    register_ptr_to_python<PeakProfilePtr>();
}

}   // namespace srrealmodule

// End of file
