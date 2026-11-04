**Added:**

* Publish self-contained wheels to PyPI for Linux (x86_64),
  macOS (x86_64, arm64) and Windows (AMD64). The wheels link
  Boost.Serialization and GSL statically and install without conda.
  ``vcpkg.json`` pins the library versions on all platforms.
* Add the ``SRREAL_STATIC_DEPS`` CMake option to link Boost.Serialization
  and GSL statically. It is off by default, so conda and system builds
  keep linking the shared libraries.

**Changed:**

* <news item>

**Deprecated:**

* <news item>

**Removed:**

* <news item>

**Fixed:**

* Include libdiffpy's ``LICENSE.rst`` and ``LICENSE_DANSE.rst`` in the
  sdist under their actual file names.

**Security:**

* <news item>
