#######
|title|
#######

.. |title| replace:: diffpy.srreal documentation

``diffpy.srreal`` - Calculators for PDF, bond valence sum, and other quantities based on atom pair interaction.

| Software version |release|
| Last updated |today|.

The diffpy.srreal package provides calculators for atomic pair distribution
function (PDF), bond valence sums (BVS), atom overlaps for a hard-sphere
model, bond distances and directions up to specified maximum distance.   The
atomic structure models are represented with internal classes as non-periodic,
periodic or structures with space group symmetries.  The package provides
implicit adapters from diffpy.structure classes or from Crystal or Molecule
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

=======
Authors
=======

diffpy.srreal is developed by members of the Billinge Group at
Columbia University and at Brookhaven National Laboratory including
Pavol Juhás, Christopher L. Farrow, Simon J.L. Billinge.

For a detailed list of contributors see
https://github.com/diffpy/diffpy.srreal/graphs/contributors.

Reference
=========

If you use this program for a scientific research that leads
to publication, we ask that you acknowledge use of the program
by citing the following paper in your publication:

   diffpy.srreal Package, https://github.com/diffpy/diffpy.srreal

============
Installation
============

See the `README <https://github.com/diffpy/diffpy.srreal#installation>`_
file included with the distribution.

================
Acknowledgements
================

``diffpy.srreal`` is built and maintained with `scikit-package <https://scikit-package.github.io/scikit-package/>`_.

=================
Table of contents
=================
.. toctree::
   :titlesonly:

   license
   release
   examples
   Package API <api/diffpy.srreal>

=======
Indices
=======

* :ref:`genindex`
* :ref:`search`
