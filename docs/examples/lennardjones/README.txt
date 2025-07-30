This example shows how to use the PairQuantity base class to create
a calculator Lennard-Jones (LJ) potential.  The LJ calculator is
defined by inheriting from the PairQuantity base class and overloading
its addPairContribution method.  This can be done at both the Python
and C++ levels as shown by example files.

ljcalculator.py  -- definition of LJ calculator in Python
ljcalculator.cpp -- definition of LJ calculator in C++.  Use "make" to compile.
lj50.xyz         -- 50-atom structure with minimum LJ potential

Ref: http://www-wales.ch.cam.ac.uk/~jon/structures/LJ/tables.150.html
