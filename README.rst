|Icon| |title|_
===============

.. |title| replace:: diffpy.srreal
.. _title: https://diffpy.github.io/diffpy.srreal

.. |Icon| image:: https://avatars.githubusercontent.com/diffpy
        :target: https://diffpy.github.io/diffpy.srreal
        :height: 100px

|PyPI| |Forge| |PythonVersion| |PR|

|CI| |Codecov| |Black| |Tracking|

.. |Black| image:: https://img.shields.io/badge/code_style-black-black
        :target: https://github.com/psf/black

.. |CI| image:: https://github.com/diffpy/diffpy.srreal/actions/workflows/matrix-and-codecov-on-merge-to-main.yml/badge.svg
        :target: https://github.com/diffpy/diffpy.srreal/actions/workflows/matrix-and-codecov-on-merge-to-main.yml

.. |Codecov| image:: https://codecov.io/gh/diffpy/diffpy.srreal/branch/main/graph/badge.svg
        :target: https://codecov.io/gh/diffpy/diffpy.srreal

.. |Forge| image:: https://img.shields.io/conda/vn/conda-forge/diffpy.srreal
        :target: https://anaconda.org/conda-forge/diffpy.srreal

.. |PR| image:: https://img.shields.io/badge/PR-Welcome-29ab47ff

.. |PyPI| image:: https://img.shields.io/pypi/v/diffpy.srreal
        :target: https://pypi.org/project/diffpy.srreal/

.. |PythonVersion| image:: https://img.shields.io/pypi/pyversions/diffpy.srreal
        :target: https://pypi.org/project/diffpy.srreal/

.. |Tracking| image:: https://img.shields.io/badge/issue_tracking-github-blue
        :target: https://github.com/diffpy/diffpy.srreal/issues

Calculators for PDF, bond valence sum, and other quantities based on atom pair interaction.

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

For more information about the diffpy.srreal library, please consult our `online documentation <https://diffpy.github.io/diffpy.srreal>`_.

Citation
--------

If you use diffpy.srreal in a scientific publication, we would like you to cite this package as

        diffpy.srreal Package, https://github.com/diffpy/diffpy.srreal

Installation
------------

The preferred method is to use `Miniconda Python
<https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html>`_
and install from the "conda-forge" channel of Conda packages.

To add "conda-forge" to the conda channels, run the following in a terminal. ::

        conda config --add channels conda-forge

We want to install our packages in a suitable conda environment.
The following creates and activates a new environment named ``diffpy.srreal_env`` ::

        conda create -n diffpy.srreal_env diffpy.srreal
        conda activate diffpy.srreal_env

To confirm that the installation was successful, type ::

        python -c "import diffpy.srreal; print(diffpy.srreal.__version__)"

The output should print the latest version displayed on the badges above.

If the above does not work, you can use ``pip`` to download and install the latest release from
`Python Package Index <https://pypi.python.org>`_.
To install using ``pip`` into your ``diffpy.srreal_env`` environment, type ::

        pip install diffpy.srreal

If you prefer to install from sources, after installing the dependencies, obtain the source archive from
`GitHub <https://github.com/diffpy/diffpy.srreal/>`_. Once installed, ``cd`` into your ``diffpy.srreal`` directory
and run the following ::

        pip install .

Getting Started
---------------

You may consult our `online documentation <https://diffpy.github.io/diffpy.srreal>`_ for tutorials and API references.

Support and Contribute
----------------------

`Diffpy user group <https://groups.google.com/g/diffpy-users>`_ is the discussion forum for general questions and discussions about the use of diffpy.srreal. Please join the diffpy.srreal users community by joining the Google group. The diffpy.srreal project welcomes your expertise and enthusiasm!

If you see a bug or want to request a feature, please `report it as an issue <https://github.com/diffpy/diffpy.srreal/issues>`_ and/or `submit a fix as a PR <https://github.com/diffpy/diffpy.srreal/pulls>`_. You can also post it to the `Diffpy user group <https://groups.google.com/g/diffpy-users>`_.

Feel free to fork the project and contribute. To install diffpy.srreal
in a development mode, with its sources being directly used by Python
rather than copied to a package directory, use the following in the root
directory ::

        pip install -e .

To ensure code quality and to prevent accidental commits into the default branch, please set up the use of our pre-commit
hooks.

1. Install pre-commit in your working environment by running ``conda install pre-commit``.

2. Initialize pre-commit (one time only) ``pre-commit install``.

Thereafter your code will be linted by black and isort and checked against flake8 before you can commit.
If it fails by black or isort, just rerun and it should pass (black and isort will modify the files so should
pass after they are modified). If the flake8 test fails please see the error messages and fix them manually before
trying to commit again.

Improvements and fixes are always appreciated.

Before contributing, please read our `Code of Conduct <https://github.com/diffpy/diffpy.srreal/blob/main/CODE-OF-CONDUCT.rst>`_.

Contact
-------

For more information on diffpy.srreal please visit the project `web-page <https://diffpy.github.io/>`_ or email Simon Billinge at sb2896@columbia.edu.

Acknowledgements
----------------

``diffpy.srreal`` is built and maintained with `scikit-package <https://scikit-package.github.io/scikit-package/>`_.
