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
* Bindings to the PDFEnvelope class.  The business methods can be overloaded
* from Python to create custom PDF envelope functions.
*
*****************************************************************************/

#include <boost/python/class.hpp>
#include <boost/python/def.hpp>
#include <boost/python/copy_const_reference.hpp>
#include <boost/python/register_ptr_to_python.hpp>

#include <diffpy/srreal/PDFEnvelope.hpp>
#include <diffpy/srreal/QResolutionEnvelope.hpp>
#include <diffpy/srreal/ScaleEnvelope.hpp>
#include <diffpy/srreal/SphericalShapeEnvelope.hpp>
#include <diffpy/srreal/StepCutEnvelope.hpp>

#include "srreal_converters.hpp"
#include "srreal_pickling.hpp"

namespace srrealmodule {
namespace nswrap_PDFEnvelope {

using namespace boost;
using namespace boost::python;
using namespace diffpy::srreal;

// docstrings ----------------------------------------------------------------

const char* doc_PDFEnvelope = "\
Base class and registry for functions that return PDF scaling envelopes.\n\
";

const char* doc_PDFEnvelope_create = "\
Return a new instance of the same type as self.\n\
\n\
This method must be overloaded in a derived class.\n\
";

const char* doc_PDFEnvelope_clone = "\
Return a new instance that is a copy of self.\n\
\n\
This method must be overloaded in a derived class.\n\
";

const char* doc_PDFEnvelope_type = "\
Return a unique string type that identifies a PDFEnvelope-derived class.\n\
The string type is used for class registration and in the createByType\n\
function.\n\
\n\
This method must be overloaded in a derived class.\n\
";

const char* doc_PDFEnvelope___call__ = "\
Calculate PDF envelope at the specified r.\n\
\n\
r    -- atom distance in Angstroms where the scale envelope is evaluated.\n\
\n\
Return float.\n\
";

const char* doc_PDFEnvelope__registerThisType = "\
Add this class to the global registry of PDFEnvelope types.\n\
\n\
This method must be called once after definition of the derived\n\
class to support pickling and the createByType factory.\n\
";

const char* doc_PDFEnvelope_createByType = "\
Return a new PDFEnvelope instance of the specified string type.\n\
\n\
tp   -- string type identifying a registered PDFEnvelope class\n\
        See getRegisteredTypes for the allowed values.\n\
\n\
Return a new instance of the PDFEnvelope-derived class.\n\
";

const char* doc_PDFEnvelope_getRegisteredTypes = "\
Return a set of string types of the registered PDFEnvelope classes.\n\
These are the allowed arguments for the createByType factory.\n\
";

const char* doc_QResolutionEnvelope = "\
Gaussian dampening envelope function due to limited Q-resolution.\n\
\n\
Returns   exp(-qdamp * x**2).\n\
Returns   1  when qdamp is zero or negative.\n\
";

const char* doc_ScaleEnvelope = "\
Uniform scaling envelope function.\n\
\n\
Returns   scale.\n\
";

const char* doc_SphericalShapeEnvelope = "\
Dampening PDF envelope due to finite spherical particle shape.\n\
\n\
Returns   (1 - 1.5*tau + 0.5*tau**3), where tau = x/spdiameter.\n\
Returns   1 when spdiameter <= 0.\n\
";

const char* doc_StepCutEnvelope = "\n\
Step function damping envelope.\n\
\n\
Returns   1 when x <= stepcut or if stepcut <= 0.\n\
Returns   0 when x > stepcut.\n\
";

// wrappers ------------------------------------------------------------------

DECLARE_PYSET_FUNCTION_WRAPPER(PDFEnvelope::getRegisteredTypes,
        getPDFEnvelopeTypes_asset)

// Helper class allows overload of the PDFEnvelope methods from Python.

class PDFEnvelopeWrap :
    public PDFEnvelope,
    public wrapper_srreal<PDFEnvelope>
{
    public:

        // HasClassRegistry methods

        PDFEnvelopePtr create() const
        {
            object rv = this->get_pure_virtual_override("create")();
            return mconfigurator.fetch(rv);
        }

        PDFEnvelopePtr clone() const
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

        double operator()(const double& x) const
        {
            return this->get_pure_virtual_override("__call__")(x);
        }

    protected:

        // HasClassRegistry method

        void setupRegisteredObject(PDFEnvelopePtr p) const
        {
            mconfigurator.setup(p);
        }

    private:

        mutable std::string mtype;
        wrapper_registry_configurator<PDFEnvelope> mconfigurator;

};  // class PDFEnvelopeWrap


std::string envelope_tostring(PDFEnvelopePtr obj)
{
    return diffpy::serialization_tostring(obj);
}


PDFEnvelopePtr envelope_fromstring(std::string content)
{
    PDFEnvelopePtr rv;
    diffpy::serialization_fromstring(rv, content);
    return rv;
}

}   // namespace nswrap_PDFEnvelope

// Wrapper definition --------------------------------------------------------

void wrap_PDFEnvelope()
{
    using namespace nswrap_PDFEnvelope;
    using diffpy::Attributes;
    namespace bp = boost::python;

    class_<PDFEnvelopeWrap, bases<Attributes>,
        noncopyable>("PDFEnvelope", doc_PDFEnvelope)
        .def("create", &PDFEnvelope::create,
                doc_PDFEnvelope_create)
        .def("clone", &PDFEnvelope::clone,
                doc_PDFEnvelope_clone)
        .def("type", &PDFEnvelope::type,
                return_value_policy<copy_const_reference>(),
                doc_PDFEnvelope_type)
        .def("__call__", &PDFEnvelope::operator(),
                bp::arg("r"), doc_PDFEnvelope___call__)
        .def("_registerThisType", &PDFEnvelope::registerThisType,
                doc_PDFEnvelope__registerThisType)
        .def("createByType", &PDFEnvelope::createByType,
                bp::arg("tp"), doc_PDFEnvelope_createByType)
        .staticmethod("createByType")
        .def("getRegisteredTypes", getPDFEnvelopeTypes_asset,
                doc_PDFEnvelope_getRegisteredTypes)
        .staticmethod("getRegisteredTypes")
        .enable_pickling()
        ;

    register_ptr_to_python<PDFEnvelopePtr>();

    class_<QResolutionEnvelope, bases<PDFEnvelope> >(
            "QResolutionEnvelope", doc_QResolutionEnvelope);
    class_<ScaleEnvelope, bases<PDFEnvelope> >(
            "ScaleEnvelope", doc_ScaleEnvelope);
    class_<SphericalShapeEnvelope, bases<PDFEnvelope> >(
            "SphericalShapeEnvelope", doc_SphericalShapeEnvelope);
    class_<StepCutEnvelope, bases<PDFEnvelope> >(
            "StepCutEnvelope", doc_StepCutEnvelope);

    // pickling support functions
    def("_PDFEnvelope_tostring", envelope_tostring);
    def("_PDFEnvelope_fromstring", envelope_fromstring);

}

}   // namespace srrealmodule

// End of file
