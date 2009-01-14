#include <boost/python.hpp>
#include <boost/python/module.hpp>
#include <boost/python/suite/indexing/map_indexing_suite.hpp>

#include <string>
#include <vector>
#include <iostream>

#include "converters.h"
#include <numpy/arrayobject.h>

#include "CrystVector/CrystVector.h"
#include "ObjCryst/Crystal.h"
#include "ObjCryst/ScatteringPower.h"
#include "ObjCryst/SpaceGroup.h"


BOOST_PYTHON_MODULE(_registerconverters)
{
    import_array();
}
