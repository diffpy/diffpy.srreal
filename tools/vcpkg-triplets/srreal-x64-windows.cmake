# Static, release-only libraries for diffpy.srreal PyPI wheels.
set(VCPKG_TARGET_ARCHITECTURE x64)
# Shared C runtime (/MD), as CPython extension modules require.
set(VCPKG_CRT_LINKAGE dynamic)
set(VCPKG_LIBRARY_LINKAGE static)
set(VCPKG_BUILD_TYPE release)
