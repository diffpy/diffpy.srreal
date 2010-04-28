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
* Bindings to the PairQuantity class.  The business protected methods
* can be overloaded from Python to create custom calculator.
* The class provides bindings to the eval and value methods for all derived
* calculators and also the double attributes access that is inherited from
* the Attributes wrapper in wrap_Attributes.
*
* Exported classes in Python:
*
* class QuantityType -- wrapped std::vector<double>
*
* class BasePairQuantity_ext -- base class to all calculators in Python

* class PairQuantity_ext -- derived class with publicized protected methods
* _addPairContribution, _resetValue, etc.  Allows their overload from Python.
*
* $Id$
*
*****************************************************************************/

#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

#include <diffpy/srreal/PairQuantity.hpp>
#include <diffpy/srreal/BaseBondGenerator.hpp>
#include <diffpy/srreal/PythonStructureAdapter.hpp>

#include "srreal_converters.hpp"

namespace srrealmodule {
namespace nswrap_PairQuantity {

using namespace boost;
using namespace boost::python;
using namespace diffpy::srreal;

// docstrings ----------------------------------------------------------------

const char* doc_BasePairQuantity_value = "\
Return total internal contributions as numpy array.\n\
";

const char* doc_BasePairQuantity_eval = "\
Calculate a pair quantity for the specified structure.\n\
\n\
stru -- structure object that can be converted to StructureAdapter\n\
\n\
Return a copy of the internal total contributions.\n\
May need to be further transformed to get the desired value.\n\
";

const char* doc_PairQuantityWrap__value = "\
Reference to the internal vector of total contributions.\n\
";

// wrappers ------------------------------------------------------------------

DECLARE_PYARRAY_METHOD_WRAPPER(value, value_asarray)

// PairQuantity::eval is a template non-constant method and
// needs an explicit wrapper function.

python::object eval_asarray(PairQuantity& obj, const python::object& a)
{
    python::object rv = convertToNumPyArray(obj.eval(a));
    return rv;
}


// representation of QuantityType objects
python::object repr_QuantityType(const QuantityType& v)
{
    python::object rv = python::str("QuantityType%r") %
        python::make_tuple(python::tuple(v));
    return rv;
}


// Helper C++ class for publicizing the protected methods.

class PairQuantityExposed : public PairQuantity
{
    public:

        // non-constant version suitable for exposing mvalue in Python
        QuantityType& value()
        {
            return mvalue;
        }


        void resizeValue(size_t sz)
        {
            this->PairQuantity::resizeValue(sz);
        }


        void resetValue()
        {
            this->PairQuantity::resetValue();
        }


        void configureBondGenerator(BaseBondGenerator& bnds)
        {
            this->PairQuantity::configureBondGenerator(bnds);
        }


        void addPairContribution(const BaseBondGenerator& bnds, int sumscale)
        {
            this->PairQuantity::addPairContribution(bnds, sumscale);
        }
};


// The second helper class allows overload of the exposed PairQuantity
// methods from Python.

class PairQuantityWrap :
    public PairQuantityExposed,
    public wrapper<PairQuantityExposed>
{
    public:

        // Make the protected virtual methods public so they
        // can be exported to Python and overloaded as well.

        void resizeValue(size_t sz)
        {
            override f = this->get_override("_resizeValue");
            if (f)  f(sz);
            else    this->default_resizeValue(sz);
        }

        void default_resizeValue(size_t sz)
        {
            this->PairQuantityExposed::resizeValue(sz);
        }


        void resetValue()
        {
            override f = this->get_override("_resetValue");
            if (f)  f();
            else    this->default_resetValue();
        }

        void default_resetValue()
        {
            this->PairQuantityExposed::resetValue();
        }


        void configureBondGenerator(BaseBondGenerator& bnds)
        {
            override f = this->get_override("_configureBondGenerator");
            if (f)  f(ptr(&bnds));
            else    this->default_configureBondGenerator(bnds);
        }

        void default_configureBondGenerator(BaseBondGenerator& bnds)
        {
            this->PairQuantityExposed::configureBondGenerator(bnds);
        }


        void addPairContribution(const BaseBondGenerator& bnds,
                int summationscale)
        {
            override f = this->get_override("_addPairContribution");
            if (f)  f(ptr(&bnds), summationscale);
            else    this->default_addPairContribution(bnds, summationscale);
        }

        void default_addPairContribution(const BaseBondGenerator& bnds,
                int summationscale)
        {
            this->PairQuantityExposed::addPairContribution(bnds, summationscale);
        }

};  // class PairQuantityWrap

}   // namespace nswrap_PairQuantity

// Wrapper definition --------------------------------------------------------

void wrap_PairQuantity()
{
    using namespace nswrap_PairQuantity;
    using diffpy::Attributes;

    class_<QuantityType>("QuantityType")
        .def(vector_indexing_suite<QuantityType>())
        .def("__repr__", repr_QuantityType)
        ;

    class_<PairQuantity, bases<Attributes> >("BasePairQuantity_ext")
        .def("value", value_asarray<PairQuantity>, doc_BasePairQuantity_value)
        .def("eval", eval_asarray, doc_BasePairQuantity_eval)
        ;

    class_<PairQuantityWrap, bases<PairQuantity>,
        noncopyable>("PairQuantity_ext")
        .def("_resizeValue",
                &PairQuantityExposed::resizeValue,
                &PairQuantityWrap::default_resizeValue)
        .def("_resetValue",
                &PairQuantityExposed::resetValue,
                &PairQuantityWrap::default_resetValue)
        .def("_configureBondGenerator",
                &PairQuantityExposed::configureBondGenerator,
                &PairQuantityWrap::default_configureBondGenerator)
        .def("_addPairContribution",
                &PairQuantityExposed::addPairContribution,
                &PairQuantityWrap::default_addPairContribution)
        .add_property("_value", make_function(&PairQuantityWrap::value,
                    return_internal_reference<>()),
                doc_PairQuantityWrap__value)
        ;
}

}   //  namespace srrealmodule

// End of file
