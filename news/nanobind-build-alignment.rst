**Changed:**

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

**Fixed:**

* Allow structureadapter to be imported before other srreal modules.
* Support the custom virtual-method dispatch used by both nanobind 2 and 3.
* Fail pyobjcryst integration tests on conversion errors instead of skipping
  them as if ObjCryst support were unavailable.
* Reject mismatched libdiffpy revisions when a local Git checkout is supplied
  for an offline build, while preserving Git-free archive builds.
