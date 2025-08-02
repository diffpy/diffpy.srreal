=============
Release notes
=============

.. current developments

1.4.0
=====

**Added:**

* Update linelength to sk-package standard of 79 characters.

**Changed:**

* Renamed CODE_OF_CONDUCTS.rst to CODE-OF-CONDUCTS.rst, doc folder to docs, and requirements/test.txt to requirements/tests.txt

**Fixed:**

* Configure ``black`` to have a linelength requirement of 79 characters.
* Update requirements to use full release `libdiffpy` and remove wrong build dependencies.
* Support ``scikit-package`` Level 5 standard (https://scikit-package.github.io/scikit-package/).



Version 1.3.0  2019-03-13
=========================

Main differences from version 1.2.

**Added:**

* Support for Python 3.7, 3.6, 3.5 in addition to 2.7.
* Validation of compiler options from `python-config`.
* Make scons scripts compatible with Python 3 and Python 2.
* `ConstantPeakWidth` attributes `uisowidth`, `bisowidth` to ease
  PDF simulation with uniform isotropic atom displacements.

**Changed:**

* Require libdiffpy 1.4 or later.
* Build Anaconda package with Anaconda C++ compiler.
* Allow language standard c++11.
* Pickle format for `PDFCalculator`, `DebyePDFCalculator`,
  `OverlapCalculator`, `PeakWidthModel`, `PeakProfile`, `PDFEnvelope`,
  `PDFBaseline`, and `ScatteringFactorTable` objects.

**Deprecated:**

* Variable `__gitsha__` in the `version` module renamed to `__git_commit__`.
* `libdiffpy_version_info` attribute `git_sha` renamed to `git_commit`.

**Removed**

* Unused method `BVParam.__hash__`.
* Disable pickling of `BasePairQuantity` as it is in effect abstract.
* Pickling of Python-added attributes to exported C++ classes.
* Function `get_libdiffpy_version_info` from the `version` module.

**Fixed**

* Return value conversion of `CrystalStructureAdapter` methods
  `expandLatticeAtom` and `getEquivalentAtoms` methods.
  Make them return a `list` of `Atom` objects.
* Name suffix resolution of `boost_python` shared library.
