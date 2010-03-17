/***********************************************************************
* $Id$
*
* Boost.python bindings to BondWidthCalculator. 
***********************************************************************/
#include "bondwidthcalculator.h"

#include <boost/utility.hpp>
#include <boost/python.hpp>
#include <boost/python/class.hpp>
#include <boost/python/module.hpp>
#include <boost/python/def.hpp>

using namespace boost::python;
using namespace SrReal;

namespace {

class BondWidthCalculatorWrap 
    : public BondWidthCalculator, public wrapper<BondWidthCalculator>
{

    public: 

    BondWidthCalculatorWrap()
        : BondWidthCalculator()
        {}

    // Do not wrap. Workaround for missing copy consructor in RefinableObj.
    BondWidthCalculatorWrap(const BondWidthCalculatorWrap& other) {}

    float default_calculate(BondPair& bp)
    { return BondWidthCalculator::calculate(bp); }

    float calculate(BondPair& bp)
    {
        if (override calculate 
                = this->get_override("calculate")) 
        {
            return calculate(bp);
        }
        return default_calculate(bp);
    }
}; // BondWidthCalculatorWrap
   
} // anonymous namespace


BOOST_PYTHON_MODULE(_bondwidthcalculator)
{

    class_<BondWidthCalculatorWrap, boost::noncopyable, 
        bases<ObjCryst::RefinableObj> >("BondWidthCalculator")
        .def("calculate", &BondWidthCalculator::calculate, 
            &BondWidthCalculatorWrap::default_calculate)
        ;

    class_<JeongBWCalculator, bases<BondWidthCalculator> >("JeongBWCalculator")
        .def("getDelta1", &JeongBWCalculator::getDelta1)
        .def("getDelta2", &JeongBWCalculator::getDelta2)
        .def("getQbroad", &JeongBWCalculator::getQbroad)
        .def("setDelta1", &JeongBWCalculator::setDelta1)
        .def("setDelta2", &JeongBWCalculator::setDelta2)
        .def("setQbroad", &JeongBWCalculator::setQbroad)
        ;
}
