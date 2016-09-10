.. image:: https://travis-ci.org/diffpy/diffpy.srreal.svg?branch=master
   :target: https://travis-ci.org/diffpy/diffpy.srreal

.. image:: http://codecov.io/github/diffpy/diffpy.srreal/coverage.svg?branch=master
   :target: http://codecov.io/github/diffpy/diffpy.srreal?branch=master

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

The diffpy.srreal requires Python 2.7, C++ compiler and
the following software:

* ``setuptools`` - tools for installing Python packages
* ``NumPy`` - library for scientific computing with Python
* ``scons`` - software constructions tool (1.0 or later)
* ``python-dev`` - header files for interfacing Python with C
* ``libboost-all-dev`` - Boost C++ libraries and development files (1.43 or later)
* ``libdiffpy`` - C++ library for PDF, bond valence sum and other pair
  quantity calculators https://github.com/diffpy/libdiffpy
* ``diffpy.Structure`` - simple storage and manipulation of atomic structures
  https://github.com/diffpy/diffpy.Structure

Optional software:

* ``periodictable`` - periodic table of elements in Python
  http://www.reflectometry.org/danse/elements.html
* ``pyobjcryst`` - Python bindings to ObjCryst++, the Object Oriented
  Crystallographic library for C++, https://github.com/diffpy/pyobjcryst.

We recommend to use `Anaconda Python <https://www.continuum.io/downloads>`_
as it allows to install all software dependencies together with
diffpy.srreal.  For other Python distributions it is necessary to
install the required software separately.  As an example, on Ubuntu
Linux some of the required software can be installed using ::

   sudo apt-get install \
      python-setuptools python-numpy scons \
      build-essential python-dev libboost-all-dev

To install the remaining packages see the installation instructions
at their respective web pages.


INSTALLATION
------------------------------------------------------------------------

The preferred method is to use Anaconda Python and install from the
"diffpy" channel of Anaconda packages ::

   conda config --add channels diffpy
   conda install diffpy.srreal

diffpy.srreal is also included in the "diffpy-cmi" collection
of packages for structure analysis ::

   conda install diffpy-cmi

If you prefer to install from sources, make sure all required software
packages are in place and then run ::

   python setup.py install

You may need to use ``sudo`` with system Python so the process is
allowed to copy files to the system directories.  If administrator (root)
access is not available, see the output from
``python setup.py install --help`` for options to install to
a user-writable location.  The installation integrity can be verified by
changing to the HOME directory and running ::

   python -m diffpy.srreal.tests.run

An alternative way of installing diffpy.srreal is to use the SCons tool,
which can speed up the process by compiling the C++ files in parallel (-j4) ::

   sudo scons -j4 install

See ``scons -h`` for decription of build targets and options.


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

When developing with Anaconda Python it is essential to specify
header path, library path and runtime library path for the active
Anaconda environment.  This can be achieved by setting the ``CPATH``,
``LIBRARY_PATH`` and ``LDFLAGS`` environment variables as follows::

   # resolve the prefix directory P of the active Anaconda environment
   P="$(conda info --json | grep default_prefix | cut -d\" -f4)"
   export CPATH=$P/include
   export LIBRARY_PATH=$P/lib
   export LDFLAGS=-Wl,-rpath,$P/lib
   # compile and re-install diffpy.srreal
   scons -j4 build=debug develop

On Mac OS X the distributed Anaconda packages are built for operating
system version 10.7, which may be incompatible with codes compiled on a
newer OS.  To avoid this problem set the environment variable
``MACOSX_DEPLOYMENT_TARGET=10.7``.  This allows to build diffpy.srreal
against the Anaconda package for the libdiffpy library.


CONTACTS
------------------------------------------------------------------------

For more information on diffpy.srreal please visit the project web-page

http://www.diffpy.org

or email Prof. Simon Billinge at sb2896@columbia.edu.
