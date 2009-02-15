/*****************************************************************************
*
* diffpy.srreal     by DANSE Diffraction group
*                   Simon J. L. Billinge
*                   (c) 2006 Trustees of the Michigan State University.
*                   All rights reserved.
*
* File coded by:    Pavol Juhas
*
* See AUTHORS.txt for a list of people who contributed.
* See LICENSE.txt for license information.
*
******************************************************************************
*
* Unit tests driver
*
* $Id$
*
*****************************************************************************/

#include <cstdlib>
#include <cppunit/CompilerOutputter.h>
#include <cppunit/extensions/TestFactoryRegistry.h>
#include <cppunit/ui/text/TestRunner.h>
#include <cppunit/extensions/HelperMacros.h>
#include <boost/python.hpp>

#include "globals.hpp"

using namespace std;
using namespace CppUnit;


int main(int argc, char* argv[])
{
    // Reset global paths from argv[0]
    thisfile(argv[0]);

    // Get the top level suite from the registry
    Test* suite = TestFactoryRegistry::getRegistry().makeTest();

    // Adds the test to the list of test to run
    TextUi::TestRunner runner;
    runner.addTest(suite);

    // Change the default outputter to a compiler error format outputter

    Outputter* outfmt = new CompilerOutputter(&runner.result(), cerr);
    runner.setOutputter(outfmt);
    // Run the tests.
    int exit_code;
    try {
        bool wasSucessful;
        wasSucessful = runner.run();
        exit_code = wasSucessful ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    catch (boost::python::error_already_set) {
        if (PyErr_Occurred())   PyErr_Print();
        exit_code = 2;
    }

    // Return error code 1 if the one of test failed.
    // Return error code 2 if there was other failure.
    return exit_code;
}

// End of file
