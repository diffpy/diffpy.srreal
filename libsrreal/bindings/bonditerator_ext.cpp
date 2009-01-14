/***********************************************************************
* $Id$
*
* Boost.python bindings to BondWidthCalculator. 
***********************************************************************/
#include "bonditerator.h"
#include "converters.h"

#include <boost/utility.hpp>
#include <boost/python.hpp>
#include <boost/python/class.hpp>
#include <boost/python/module.hpp>
#include <boost/python/def.hpp>
#include <boost/python/list.hpp>

#include <string>
#include <iostream>

using namespace boost::python;
using namespace SrReal;

namespace {

float getX(ShiftedSC &sc)
{
    return sc.xyz[0];
}

void setX(ShiftedSC &sc, float val)
{
    sc.xyz[0] = val;
}

float getY(ShiftedSC &sc)
{
    return sc.xyz[1];
}

void setY(ShiftedSC &sc, float val)
{
    sc.xyz[2] = val;
}

float getZ(ShiftedSC &sc)
{
    return sc.xyz[2];
}

void setZ(ShiftedSC &sc, float val)
{
    sc.xyz[2] = val;
}

list getUnitCellFromCrystal(ObjCryst::Crystal &crystal)
{
    std::vector<ShiftedSC> unitcell = getUnitCell(crystal);

    list uclist;
    for(size_t i=0; i<unitcell.size(); i++)
    {
        uclist.append(unitcell[i]);
    }
    return uclist;
}

list getUnitCellFromBondIterator(BondIterator& bonditer)
{
    std::vector<ShiftedSC> unitcell = bonditer.getUnitCell();

    list uclist;
    for(size_t i=0; i<unitcell.size(); i++)
    {
        uclist.append(unitcell[i]);
    }
    return uclist;
}

} // anonymous namespace


BOOST_PYTHON_MODULE(_bonditerator)
{

    def("getUnitCell", &getUnitCellFromCrystal);

    class_<ShiftedSC>("ShiftedSC", init<>())
        .def(init<const ShiftedSC&>())
        .def(init<const ObjCryst::ScatteringComponent*,
            const float, const float, const float, const int>())
        .def_readwrite("sc", &ShiftedSC::sc)
        .def_readwrite("id", &ShiftedSC::id)
        .add_property("x", &getX, &setX)
        .add_property("y", &getY, &setY)
        .add_property("z", &getZ, &setZ)
        .def(self < self)
        .def(self == self)
        // FIXME - this one doesn't work
        //.def(self_ns::str(self)) // seg-faults
        ;

    class_<BondPair>("BondPair", init<>())
        .def("getXYZ1", (float (BondPair::*)(size_t)) &BondPair::getXYZ1)
        .def("getXYZ2", (float (BondPair::*)(size_t)) &BondPair::getXYZ2)
        .def("setXYZ1", (void (BondPair::*)(size_t, float)) &BondPair::setXYZ1)
        .def("setXYZ2", (void (BondPair::*)(size_t, float)) &BondPair::setXYZ2)
        .def("setSC1", &BondPair::setSC1)
        .def("getSC1", &BondPair::getSC1, return_internal_reference<>())
        .def("setSC2", &BondPair::setSC2)
        .def("getSC2", &BondPair::getSC2, return_internal_reference<>())
        .def("getMultiplicity", &BondPair::getMultiplicity)
        .def("setMultiplicity", &BondPair::setMultiplicity)
        .def("getDistance", &BondPair::getDistance)
        // FIXME - this one doesn't work
        //.def(str(self))
        //.def(self_ns::str(self)) // seg-faults
        ;

    class_<BondIterator>("BondIterator", init<const BondIterator&>())
        .def(init<ObjCryst::Crystal&>())
        .def(init<ObjCryst::Crystal&,float,float>())
        .def("setBondRange", &BondIterator::setBondRange)
        .def("setScatteringComponent", &BondIterator::setScatteringComponent)
        .def("rewind", &BondIterator::rewind)
        .def("next", &BondIterator::next)
        .def("finished", &BondIterator::finished)
        .def("getBondPair", &BondIterator::getBondPair)
        .def("getRmin", &BondIterator::getRmin)
        .def("getRmax", &BondIterator::getRmax)
        .def("getCrystal", &BondIterator::getCrystal, 
                return_internal_reference<>())
        .def("getUnitCell", &getUnitCellFromBondIterator)
        ;
}
