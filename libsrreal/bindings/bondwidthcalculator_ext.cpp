/***********************************************************************
* $Id$
*
* Boost.python bindings to BondWidthCalculator. 
***********************************************************************/
#include "bondwidthcalculator.h"
#include "converters.h"

#include <boost/utility.hpp>
#include <boost/python.hpp>
#include <boost/python/class.hpp>
#include <boost/python/module.hpp>
#include <boost/python/def.hpp>

#include <string>
#include <iostream>

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
   
class JeongBWCalculatorWrap 
    : public JeongBWCalculator, public wrapper<JeongBWCalculator>
{

    public: 

    JeongBWCalculatorWrap() : JeongBWCalculator() {}

    float default_calculate(BondPair& bp)
    { return JeongBWCalculator::calculate(bp); }

    float calculate(BondPair& bp)
    {
        if (override calculate 
                = this->get_override("calculate")) 
        {
            return calculate(bp);
        }
        return default_calculate(bp);
    }
}; // JeongBWCalculator

} // anonymous namespace


BOOST_PYTHON_MODULE(_bondwidthcalculator)
{

    class_<BondWidthCalculatorWrap, boost::noncopyable, 
        bases<ObjCryst::RefinableObj> >("BondWidthCalculator")
        .def("calculate", &BondWidthCalculator::calculate, 
            &BondWidthCalculatorWrap::default_calculate)
        ;

    class_<JeongBWCalculatorWrap, boost::noncopyable, 
        bases<ObjCryst::RefinableObj> >("JeongBWCalculator")
        .def("calculate", &JeongBWCalculator::calculate, 
            &JeongBWCalculatorWrap::default_calculate)
        ;
}
