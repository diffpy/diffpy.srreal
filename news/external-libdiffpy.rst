**Changed:**

* Build the Python extension against an independently installed libdiffpy
  with the C++23 ``std::shared_ptr`` API. Source builds no longer download
  or compile libdiffpy, and distributions no longer include its runtime data.

**Fixed:**

* Disambiguate tuple construction when restoring pickles with MSVC.
