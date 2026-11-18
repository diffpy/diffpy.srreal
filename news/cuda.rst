**Added:**

* The ``diffpy.srreal.calculationkernel`` module selects the kernel of PDF
  and Debye calculations.
* ``AtomicStructureAdapter.getAtomArrays`` and ``setAtomArrays`` read and
  replace all atom data in bulk.

**Changed:**

* Full PDF and Debye calculations use multithreaded, batched CPU sums, with
  SIMD kernels when built with xsimd (always in wheels).
* ``OPTIMIZED`` can choose a full calculation when it costs less than updating
  changed atoms. ``evaluatortypeused`` then reports ``"BASIC"``.
* Conversion of ``diffpy.structure`` and ``pyobjcryst`` structures is faster.

**Deprecated:**

* <news item>

**Removed:**

* <news item>

**Fixed:**

* ``DebyePDFCalculator`` uses each site's own occupancy when several sites
  share an element.
* ``CrystalStructureAdapter.getEquivalentAtoms`` now checks the index against
  the number of sites, fixing incorrect exceptions and a possible crash.
* ``OPTIMIZED`` recalculates when a change of adapter class changes the
  structure, for example when a cluster becomes periodic.

**Security:**

* <news item>
