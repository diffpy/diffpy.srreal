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
* Bindings to the PDFEnvelope class.  The business methods can be overloaded
* from Python to create custom PDF envelope functions.
*
* $Id$
*
*****************************************************************************/

#include <boost/python.hpp>

#include <diffpy/srreal/PDFEnvelope.hpp>

#include "srreal_converters.hpp"

namespace srrealmodule {
namespace nswrap_PDFEnvelope {

using namespace boost;
using namespace boost::python;
using namespace diffpy::srreal;

// Helper class allows overload of the PDFEnvelope methods from Python.

class PDFEnvelopeWrap :
    public PDFEnvelope,
    public wrapper<PDFEnvelope>
{
    public:

        // constructors

        boost::shared_ptr<PDFEnvelope> create() const
        {
            return this->get_override("_create")();
        }

        boost::shared_ptr<PDFEnvelope> clone() const
        {
            return this->get_override("_clone")();
        }

        // methods

        const std::string& type() const
        {
            return this->get_override("type")();
        }

        double operator()(const double& x) const
        {
            return this->get_override("__call__")(x);
        }

};  // class PDFEnvelopeWrap

}   // namespace nswrap_PDFEnvelope

// Wrapper definition --------------------------------------------------------

void wrap_PDFEnvelope()
{
    using namespace nswrap_PDFEnvelope;
    using diffpy::Attributes;

    class_<PDFEnvelopeWrap, noncopyable,
        bases<Attributes> >("PDFEnvelope_ext")
        .def("_create", pure_virtual(&PDFEnvelope::create))
        .def("_clone", pure_virtual(&PDFEnvelope::clone))
        .def("type", pure_virtual(&PDFEnvelope::type),
                return_value_policy<copy_const_reference>())
        .def("__call__", pure_virtual(&PDFEnvelope::operator()))
        ;
}

}   // namespace srrealmodule

// End of file
