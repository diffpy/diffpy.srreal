#!/bin/bash

export CPATH="${PREFIX}/include:$CPATH"
export LIBRARY_PATH="${PREFIX}/lib:$LIBRARY_PATH"

if [ `uname` == Darwin ]; then
    export DYLD_FALLBACK_LIBRARY_PATH="${PREFIX}/lib"
else
    export LD_LIBRARY_PATH="${PREFIX}/lib"
fi

scons -j $CPU_COUNT install prefix=$PREFIX

# Add more build steps here, if they are necessary.

$PYTHON setup.py --version > __conda_version__.txt

# See
# http://docs.continuum.io/conda/build.html
# for a list of environment variables that are set during the build process.
