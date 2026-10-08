include("${CMAKE_CURRENT_LIST_DIR}/LibdiffpyVersion.cmake")
include(FetchContent)

# Use extraction time so changing the archive also rebuilds its sources.
if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

FetchContent_Declare(libdiffpy
    URL "https://codeload.github.com/diffpy/libdiffpy/tar.gz/${DIFFPY_GIT_SHA}"
    URL_HASH "SHA256=${LIBDIFFPY_ARCHIVE_SHA256}"
    TLS_VERIFY TRUE
)
# libdiffpy uses SCons; we compile its sources in our own CMake targets.
FetchContent_MakeAvailable(libdiffpy)
set(LIBDIFFPY_SOURCE_DIR "${libdiffpy_SOURCE_DIR}")

if(NOT EXISTS "${LIBDIFFPY_SOURCE_DIR}/src/diffpy/version.tpl")
    message(FATAL_ERROR "Missing libdiffpy sources in ${LIBDIFFPY_SOURCE_DIR}.")
endif()

# Also validate Git checkouts supplied through FETCHCONTENT_SOURCE_DIR_LIBDIFFPY.
include("${CMAKE_CURRENT_LIST_DIR}/VerifyLibdiffpyRevision.cmake")
verify_libdiffpy_revision("${LIBDIFFPY_SOURCE_DIR}" "${DIFFPY_GIT_SHA}")
