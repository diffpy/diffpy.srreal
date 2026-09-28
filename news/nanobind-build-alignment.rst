**Changed:**

* Align the nanobind migration with pyobjcryst issue 101: use scikit-build-core,
  CMake, and pinned upstream libdiffpy sources compiled into the extension.
  Retain C++23 to match libdiffpy's own build requirements.
* Use pyobjcryst's Python API for Crystal and Molecule conversion, removing
  the build-time libobjcryst dependency and cross-extension C++ ABI coupling.
* Build and test source distributions and repaired wheels on Linux and macOS
  against the current pyobjcryst migration branch, recording each resolved
  revision. Preserve tag-based package versions.

**Fixed:**

* Allow structureadapter to be imported before other srreal modules.
* Support the custom virtual-method dispatch used by both nanobind 2 and 3.
* Fail pyobjcryst integration tests on conversion errors instead of skipping
  them as if ObjCryst support were unavailable.
* Reject mismatched libdiffpy version metadata during Git checkout builds and
  before CI creates a source archive, while preserving Git-free archive builds.
