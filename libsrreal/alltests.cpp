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

using namespace std;
using namespace CppUnit;


int main()
{
    // Get the top level suite from the registry
    Test* suite = TestFactoryRegistry::getRegistry().makeTest();

    // Adds the test to the list of test to run
    TextUi::TestRunner runner;
    runner.addTest(suite);

    // Change the default outputter to a compiler error format outputter

    Outputter* outfmt = new CompilerOutputter(&runner.result(), cerr);
    runner.setOutputter(outfmt);
    // Run the tests.
    bool wasSucessful = runner.run();

    // Return error code 1 if the one of test failed.
    return wasSucessful ? EXIT_SUCCESS : EXIT_FAILURE;
}

// End of file
