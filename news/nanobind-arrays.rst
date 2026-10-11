**Changed:**

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

**Fixed:**

* Read strided and reversed integer index arrays in logical order instead of
  treating their storage as contiguous.
* Interpret byte-swapped FFT inputs and integer index arrays in their actual
  byte order, and safely handle unaligned buffers.
* Reject integer-array indices outside the C++ ``int`` range with
  ``OverflowError`` instead of silently wrapping. These three corrections
  intentionally change erroneous behavior inherited from Boost.Python.
