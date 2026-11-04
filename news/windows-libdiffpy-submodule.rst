**Changed:**

* Build the embedded libdiffpy core from a pinned Git submodule. Source
  distributions include the core sources and runtime data for offline builds.

**Fixed:**

* Use libdiffpy's Windows-compatible runtime data lookup and declarations,
  enable MSVC mathematical constants, and disambiguate tuple restoration
  in calculator pickles.
