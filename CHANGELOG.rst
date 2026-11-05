=============
Release notes
=============

.. current developments

1.5.0
=====

**Added:**

* Publish self-contained wheels to PyPI for Linux (x86_64),
macOS (x86_64, arm64) and Windows (AMD64). The wheels link
Boost.Serialization and GSL statically and install without conda.
``vcpkg.json`` pins the library versions on all platforms.
* Add the ``SRREAL_STATIC_DEPS`` CMake option to link Boost.Serialization
and GSL statically. It is off by default, so conda and system builds
keep linking the shared libraries.
* Update citation reference in README and docs.

**Changed:**

* Build the embedded libdiffpy core from a pinned Git submodule. Source
distributions include the core sources and runtime data for offline builds.
* Use typed nanobind arrays for numeric input, copied results, and writable
Atom views. Keep NumPy output types, shapes, native dtypes, and the existing
scalar-versus-array return conventions.
* Remove the NumPy C API and build-time NumPy dependency. NumPy remains a
runtime dependency; wheel Python ABI requirements are unchanged.
* Retain NumPy-owned writable storage for copied results and explicit storage
owners for Atom views, including slices retained after structural edits.
* Preserve endpoint-specific conversion rules, NumPy scalar casting warnings,
and coordinate/matrix broadcasting. Invalid coordinate/matrix assignments
no longer partially modify an Atom.
* updated cpp minimum version to c++23.
* replaced boost smart pointer usages with stl implementations.
* Added Python 3.14 support, removed Python 3.11 support.
* Align the nanobind migration with pyobjcryst issue 101: use scikit-build-core,
CMake, and pinned upstream libdiffpy sources compiled into the extension.
Retain C++23 to match libdiffpy's own build requirements.
* Use pyobjcryst's Python API for Crystal and Molecule conversion, removing
the build-time libobjcryst dependency and cross-extension C++ ABI coupling.
* Retain the centralized scikit-package testing, release, and documentation
workflows and their platform matrix. Download the pinned libdiffpy source
archive with CMake and verify its SHA-256 checksum, without a Git submodule
or unsupported shared-workflow inputs. Test the current pyobjcryst migration
on Linux through the shared post-install hook. Preserve tag-based package
versions.
* Structure conversion now returns ordinary atomic or periodic adapters,
without reading ``pdffit`` metadata. Configure PDF parameters explicitly
through the existing calculator interfaces.
* ``nometa()`` returns or converts the source adapter without a proxy.
``None`` returns the empty adapter.
* Require nanobind 3.1 or newer within the 3.x series.
* Use consistent Atom references for indexing and iteration. Surviving atoms
follow index shifts; removed or replaced atoms detach safely. NumPy views
retained across structural edits become snapshots; reacquire a view to
modify the updated structure.
* Expose calculator component properties directly without preserving the
``PeakWidthModelOwner`` and ``ScatteringFactorTableOwner`` inheritance
relationships. Use the calculator's own properties and methods.
* Use Python ``str`` for text parameters; decode byte strings explicitly.
Binary serialization payloads remain ``bytes``.
* Follow nanobind's numerical conversions. Explicitly call ``float(value)``
when an overridden ``__float__`` must be honored. ``None`` is not accepted
as numerical input to baselines, envelopes, or scattering-factor lookup.
* Use value equality and nanobind's representation for ``QuantityType``.
These mutable containers are unhashable, and slice assignment requires
an iterable, including for a single replacement value.
* Interpret ``__getstate_manages_dict__`` by its truth value. Custom pickle
hooks on guarded classes must preserve Python metadata and set this flag
to ``True``.

**Deprecated:**

* ``NoMetaStructureAdapter``, retained for saved objects and subclasses.
Its Python constructor now emits ``DeprecationWarning``. Historical
metadata adapter classes remain importable for pickle compatibility.
* Deprecated setScatteringFactorTableByType.

**Fixed:**

* Include libdiffpy's ``LICENSE.rst`` and ``LICENSE_DANSE.rst`` in the
sdist under their actual file names.
* Fixed example to pass correct Molecule structure for PDF comparison
* Use libdiffpy's Windows-compatible runtime data lookup and declarations,
enable MSVC mathematical constants, and disambiguate tuple restoration
in calculator pickles.
* Use ``Lattice.cell_parms`` in place of the deprecated ``Lattice.abcABG`` when converting ``diffpy.structure`` objects to silence deprecation warnings.
* Use ``load_structure``, ``get_parser``, ``parse_file``, ``read_structure``, ``set_latt_parms``, and ``add_new_atom`` in place of the deprecated ``diffpy.structure`` functions in tests, examples, and docstrings.
* Fixed `parallelPDF.py` example multiprocessing safe importing of main module issue.
* Install libdiffpy's ``LICENSE.rst`` and ``LICENSE_DANSE.rst`` instead of the renamed ``LICENSE.txt`` and ``LICENSE_DANSE.txt``.
* Read strided and reversed integer index arrays in logical order instead of
treating their storage as contiguous.
* Interpret byte-swapped FFT inputs and integer index arrays in their actual
byte order, and safely handle unaligned buffers.
* Reject integer-array indices outside the C++ ``int`` range with
``OverflowError`` instead of silently wrapping. These three corrections
intentionally change erroneous behavior inherited from Boost.Python.
* Allow structureadapter to be imported before other srreal modules.
* Support the custom virtual-method dispatch used by both nanobind 2 and 3.
* Fail pyobjcryst integration tests on conversion errors instead of skipping
them as if ObjCryst support were unavailable.
* Reject mismatched libdiffpy revisions when a local Git checkout is supplied
for an offline build, while preserving Git-free archive builds.
* Prevent repeated Python calculation callbacks when calling base methods.
* Restore structure sequence edits and resizing contribution-array slices.
* Keep retained Atom references and NumPy views safe during structure edits
and deserialization.
* Keep public calculator exports callable and configurable, including objects
loaded from supported Boost.Python pickles.
* Restore saved structures and Python-defined components, honor subclass
construction hooks, and guard against accidental metadata loss in custom
pickle hooks.
* Accept Python and NumPy truth values in adapter anisotropy callbacks.

**Removed:**

* Remove utils scripts ``devutils`` from the package.
* Remove the deprecated ``setScatteringFactorTableByType`` method from
``ScatteringFactorTableOwner``, ``PDFCalculator``, and ``DebyePDFCalculator``.
* The ``_customPQConfig`` bindings and Python callback dispatch, including
automatic application of structure metadata to PDF calculators.


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
