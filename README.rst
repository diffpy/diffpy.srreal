diffpy.srreal
========================================================================

Calculators for PDF, bond valence sum and other pair quantities

The diffpy.srreal package provides calculators for atomic pair distribution
function (PDF), bond valence sums (BVS), atom overlaps for a hard-sphere
model, bond distances and directions up to specified maximum distance.   The
atomic structure models are represented with internal classes as non-periodic,
periodic or structures with space group symmetries.  The package provides
implicit adapters from diffpy.Structure class or from Crystal or Molecule
objects from pyobjcryst.  Adapters can be easily defined for any other
structure representations in Python allowing their direct use with the
calculators.  Calculators support two evaluation models - BASIC, which
performs a full pair-summation every time, and OPTIMIZED, which updates only
pair contributions that have changed since the last evaluation.  Calculations
can be split among parallel jobs using Python multiprocessing package or any
other library that provides parallel map function.  PDF calculations can
be done in two modes - either as a real-space summation of peak profiles
(PDFCalculator) or as a reciprocal-space Debye summation and Fourier
transform of the total scattering structure function (DebyePDFCalculator).

The diffpy.srreal package is a Python binding to the C++ library libdiffpy
(https://github.com/diffpy/libdiffpy).  Calculators are created as
objects of a given calculator type and so multiple instances of the same
calculator type can exist with different configurations.  Calculators are
composed of other objects that perform lower-level tasks, such as calculating
peak profile or looking up atom scattering factors.  These objects can be
re-assigned at runtime allowing to easily customize the calculation procedure.
New classes can be defined using object inheritance either in Python or in C++
and used with the existing calculators; as an example, this allows to
calculate PDF with a user-defined profile function.  A new calculator class
can be also defined for any quantity that is obtained by iteration over atom
pairs, by defining only the function that processes atom-pair contributions.

For more information about the diffpy.srreal library, see users manual at
http://diffpy.github.io/diffpy.srreal.


REQUIREMENTS
------------------------------------------------------------------------

The diffpy.srreal requires Python 2.6 or 2.7 and the following software:

* ``setuptools`` - tools for installing Python packages
* ``NumPy`` - library for scientific computing with Python
* ``scons`` - software constructions tool (1.0 or later)
* ``python-dev`` - header files for interfacing Python with C
* ``libboost-dev`` - Boost C++ libraries development files (1.43 or later)
* ``libdiffpy`` - C++ library for PDF, bond valence sum and other pair
  quantity calculators https://github.com/diffpy/libdiffpy
* ``diffpy.Structure`` - simple storage and manipulation of atomic structures
  https://github.com/diffpy/diffpy.Structure

Recommended software:

* ``periodictable`` - periodic table of elements in Python
  http://www.reflectometry.org/danse/elements.html
* ``pyobjcryst`` - Python bindings to ObjCryst++, the Object Oriented
  Crystallographic library for C++, https://github.com/diffpy/pyobjcryst.

Some of the required software may be available in the system package manager,
for example, on Ubuntu Linux the dependencies can be installed as::

   sudo apt-get install \
      python-setuptools python-numpy scons \
      build-essential python-dev libboost-dev

For Mac OS X machine with the MacPorts package manager one could do ::

   sudo port install \
      python27 py27-setuptools py27-numpy scons boost

When installing with MacPorts, make sure the MacPorts bin directory is the
first in the system PATH and that python27 is selected as the default
Python version in MacPorts::

   sudo port select --set python python27

For other required packages see their respective web pages for installation
instructions.


INSTALLATION
------------------------------------------------------------------------

The easiest option is to use the latest DiffPy-CMI release bundle from
http://www.diffpy.org, which comes with diffpy.srreal and all other
dependencies included.

If you prefer to install from sources, make sure all required software
packages are in place and then run ::

   sudo python setup.py install

This installs diffpy.srreal for all users in the default system location.
If administrator (root) access is not available, see the usage info from
``python setup.py install --help`` for options to install to a user-writable
location.  The installation integrity can be verified by changing to
the HOME directory and running ::

   python -m diffpy.srreal.tests.run

An alternative way of installing diffpy.srreal is to use the SCons tool,
which can speed up the process by compiling the C++ files in parallel (-j4)::

   sudo scons -j4 install

See ``scons -h`` for build parameters and options to install to a user-writable
directory.


DEVELOPMENT
------------------------------------------------------------------------

diffpy.srreal is an open-source software developed as a part of the
DiffPy-CMI complex modeling initiative at the Brookhaven National
Laboratory.  The diffpy.srreal sources are hosted at
https://github.com/diffpy/diffpy.srreal.

Feel free to fork the project and contribute.  To install diffpy.srreal
in a development mode, where the sources are directly used by Python
rather than copied to a system directory, use ::

   python setup.py develop --user

To rebuild the C++ extension module and then optionally test the code
integrity, use ::

   scons -j4 build=debug develop [test]


CONTACTS
------------------------------------------------------------------------

For more information on diffpy.srreal please visit the project web-page

http://www.diffpy.org

or email Prof. Simon Billinge at sb2896@columbia.edu.
