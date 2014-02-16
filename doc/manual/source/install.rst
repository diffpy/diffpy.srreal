Installation
========================================================================

.. index:: Requirements

Requirements
------------------------------------------------------------------------

The diffpy.srreal requires Python 2.6 or 2.7 and the following software:

    setuptools   -- tools for installing Python packages
    NumPy        -- library for scientific computing with Python
    scons        -- software constructions tool (1.0 or later)
    python-dev   -- header files for interfacing Python with C
    libboost-dev -- Boost C++ libraries development files (1.43 or later)
    libdiffpy    -- C++ library for PDF, bond valence sum and other pair
                    quantity calculators https://github.com/diffpy/libdiffpy/
    diffpy.Structure -- simple storage and manipulation of atomic structures
                    https://github.com/diffpy/diffpy.Structure/

Recommended software:

    periodictable -- periodic table of elements in Python
                    http://www.reflectometry.org/danse/elements.html
    pyobjcryst   -- Python bindings to ObjCryst++, the Object Oriented
                    Crystallographic library for C++
                    https://github.com/diffpy/pyobjcryst/

Some of the required software may be available in the system package manager,
for example, on Ubuntu Linux the dependencies can be installed as:

    sudo apt-get install \
        python-setuptools python-numpy scons \
        build-essential python-dev libboost-dev

For Mac OS X machine with the MacPorts package manager one could do

    sudo port install \
        python27 py27-setuptools py27-numpy scons boost

When installing with MacPorts, make sure the MacPorts bin directory is the
first in the system PATH and that python27 is selected as the default
Python version in MacPorts:

    sudo port select --set python python27

For other required packages see their respective web pages for installation
instructions.


.. index:: Installation

Installation
------------------------------------------------------------------------

The easiest option is to use the latest DiffPy-CMI release bundle from
http://www.diffpy.org/, which comes with diffpy.srreal and all other
dependencies included.

If you prefer to install from sources, make sure all required software
packages are in place and then run

    sudo python setup.py install

This installs diffpy.srreal for all users in the default system location.
If administrator (root) access is not available, see the usage info from
"python setup.py install --help" for options to install to a user-writable
location.  The installation integrity can be verified by changing to
the HOME directory and running

    python -m diffpy.srreal.tests.run

An alternative way of installing diffpy.srreal is to use the scons tool,
which can speed up the process by compiling the C++ files in parallel (-j4):

    sudo scons -j4 install

See "scons -h" for build parameters and options to install to a user-writable
directory.


DEVELOPMENT

diffpy.srreal is an open-source software developed as a part of the
DiffPy-CMI complex modeling initiative at the Brookhaven National
Laboratory.  The diffpy.srreal sources are hosted at

    https://github.com/diffpy/diffpy.srreal

Feel free to fork the project and contribute.  To install diffpy.srreal
in a development mode, where the sources are directly used by Python
rather than copied to a system directory, use

    python setup.py develop --user

To rebuild the C++ extension module and then optionally test the code
integrity, use

    scons -j4 build=debug develop [test]
