#!/bin/bash

MYNCPU=$(( (CPU_COUNT > 4) ? 4 : CPU_COUNT ))

# Apply sconscript.local customizations.
cp ${RECIPE_DIR}/sconscript.local ./

# Install srreal with scons to utilize multiple CPUs.
scons -j $MYNCPU install prefix=$PREFIX
