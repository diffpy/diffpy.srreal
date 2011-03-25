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
#include <diffpy/srreal/PythonStructureAdapter.hpp>

#include "srreal_converters.hpp"
#include "srreal_pickling.hpp"

namespace srrealmodule {
namespace nswrap_PairQuantity {

using namespace boost;
using namespace boost::python;
using namespace diffpy::srreal;

// docstrings ----------------------------------------------------------------

const char* doc_BasePairQuantity_eval = "\
Calculate a pair quantity for the specified structure.\n\
\n\
stru -- structure object that can be converted to StructureAdapter.\n\
        Use the last structure when None.\n\
\n\
Return a copy of the internal total contributions.\n\
May need to be further transformed to get the desired value.\n\
";

const char* doc_BasePairQuantity_value = "\
Internal vector of total contributions as numpy array.\n\
";

const char* doc_BasePairQuantity__mergeParallelData = "\
Process raw results string from a parallel job and add them to this instance.\n\
\n\
pdata    -- raw data string from the parallel job's _getParallelData function.\n\
ncpu     -- number of parallel jobs.  The finishValue method is called after\n\
            merging ncpu parallel values.\n\
\n\
No return value.\n\
";

const char* doc_BasePairQuantity__getParallelData = "\
Return raw results string from a parallel job.\n\
";

const char* doc_PairQuantityWrap__value = "\
Reference to the internal vector of total contributions.\n\
";

// wrappers ------------------------------------------------------------------

// representation of QuantityType objects
python::object repr_QuantityType(const QuantityType& v)
{
    python::object rv = ("QuantityType%r" %
        python::make_tuple(python::tuple(v)));
    return rv;
}


// PairQuantity::eval is a template non-constant method and
// needs an explicit wrapper function.

python::object eval_asarray(PairQuantity& obj, const python::object& a)
{
    QuantityType value = (Py_None == a.ptr()) ? obj.eval() : obj.eval(a);
    python::object rv = convertToNumPyArray(value);
    return rv;
}

// support "all", "ALL" and integer iterables in setPairMask

std::vector<int> parsepairindex(python::object i)
{
    std::vector<int> rv;
    // single integer
    python::extract<int> geti(i);
    if (geti.check())
    {
        rv.push_back(geti());
        return rv;
    }
    // string equal "all" or "ALL"
    python::extract<std::string> gets(i);
    if (gets.check())
    {
        python::str lc_all(PairQuantity::ALLATOMSSTR);
        python::str uc_all = lc_all.upper();
        if (i != lc_all && i != uc_all)
        {
            python::object emsg = ("String argument must be %r or %r." %
                 python::make_tuple(lc_all, uc_all));
            PyErr_SetObject(PyExc_ValueError, emsg.ptr());
            throw_error_already_set();
        }
        rv.push_back(PairQuantity::ALLATOMSINT);
        return rv;
    }
    // sequence of integers
    python::stl_input_iterator<int> begin(i), end;
    rv.assign(begin, end);
    return rv;
}


void set_pair_mask(PairQuantity& obj,
        python::object i, python::object j, bool mask)
{
    python::extract<int> geti(i);
    python::extract<int> getj(j);
    // short circuit for normal call
    if (geti.check() && getj.check())
    {
        obj.setPairMask(geti(), getj(), mask);
        return;
    }
    std::vector<int> iindices = parsepairindex(i);
    std::vector<int> jindices = parsepairindex(j);
    std::vector<int>::const_iterator ii, jj;
    for (ii = iindices.begin(); ii != iindices.end(); ++ii)
    {
        for (jj = jindices.begin(); jj != jindices.end(); ++jj)
        {
            obj.setPairMask(*ii, *jj, mask);
        }
    }
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


        void configureBondGenerator(BaseBondGenerator& bnds) const
        {
            this->PairQuantity::configureBondGenerator(bnds);
        }


        void addPairContribution(const BaseBondGenerator& bnds, int sumscale)
        {
            this->PairQuantity::addPairContribution(bnds, sumscale);
        }


        void executeParallelMerge(const std::string& pdata)
        {
            this->PairQuantity::executeParallelMerge(pdata);
        }


        void finishValue()
        {
            this->PairQuantity::finishValue();
        }

};


// The second helper class allows overload of the exposed PairQuantity
// methods from Python.

class PairQuantityWrap :
    public PairQuantityExposed,
    public wrapper<PairQuantityExposed>
{
    public:

        // Make getParallelData overloadable from Python.

        std::string getParallelData() const
        {
            override f = this->get_override("_getParallelData");
            if (f)  return f();
            return this->default_getParallelData();
        }

        std::string default_getParallelData() const
        {
            return this->PairQuantityExposed::getParallelData();
        }

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


        void configureBondGenerator(BaseBondGenerator& bnds) const
        {
            override f = this->get_override("_configureBondGenerator");
            if (f)  f(ptr(&bnds));
            else    this->default_configureBondGenerator(bnds);
        }

        void default_configureBondGenerator(BaseBondGenerator& bnds) const
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


        void executeParallelMerge(const std::string& pdata)
        {
            override f = this->get_override("_executeParallelMerge");
            if (f)  f(pdata);
            else    this->default_executeParallelMerge(pdata);
        }

        void default_executeParallelMerge(const std::string& pdata)
        {
            this->PairQuantityExposed::executeParallelMerge(pdata);
        }


        void finishValue()
        {
            override f = this->get_override("_finishValue");
            if (f)  f();
            else    this->default_finishValue();
        }

        void default_finishValue()
        {
            this->PairQuantityExposed::finishValue();
        }

};  // class PairQuantityWrap

}   // namespace nswrap_PairQuantity

// Wrapper definition --------------------------------------------------------

void wrap_PairQuantity()
{
    using namespace nswrap_PairQuantity;
    using diffpy::Attributes;
    const python::object None;

    class_<QuantityType>("QuantityType")
        .def(vector_indexing_suite<QuantityType>())
        .def("__repr__", repr_QuantityType)
        ;

    class_<PairQuantity, bases<Attributes> >("BasePairQuantity_ext")
        .def("eval", eval_asarray, python::arg("stru")=None,
                doc_BasePairQuantity_eval)
        .add_property("value", value_asarray<PairQuantity>,
                doc_BasePairQuantity_value)
        .def("_mergeParallelData", &PairQuantity::mergeParallelData,
                (python::arg("pdata"), python::arg("ncpu")),
                doc_BasePairQuantity__mergeParallelData)
        .def("_getParallelData", &PairQuantity::getParallelData,
                doc_BasePairQuantity__getParallelData)
        .def("setStructure", &PairQuantity::setStructure<object>)
        .def("getStructure", &PairQuantity::getStructure,
                return_value_policy<copy_const_reference>())
        .def("_setupParallelRun", &PairQuantity::setupParallelRun)
        .def("maskAllPairs", &PairQuantity::maskAllPairs)
        .def("invertMask", &PairQuantity::invertMask)
        .def("setPairMask", set_pair_mask)
        .def("getPairMask", &PairQuantity::getPairMask)
        .def("setTypeMask", &PairQuantity::setTypeMask)
        .def("getTypeMask", &PairQuantity::getTypeMask)
        .def_pickle(SerializationPickleSuite<PairQuantity>())
        ;

    class_<PairQuantityWrap, bases<PairQuantity>,
        noncopyable>("PairQuantity_ext")
        .def("_getParallelData",
                &PairQuantityExposed::getParallelData,
                &PairQuantityWrap::default_getParallelData)
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
        .def("_executeParallelMerge",
                &PairQuantityExposed::executeParallelMerge,
                &PairQuantityWrap::default_executeParallelMerge)
        .def("_finishValue",
                &PairQuantityExposed::finishValue,
                &PairQuantityWrap::default_finishValue)
        .add_property("_value", make_function(&PairQuantityWrap::value,
                    return_internal_reference<>()),
                doc_PairQuantityWrap__value)
        ;

}

}   //  namespace srrealmodule

// End of file
