# Metadata for the pinned extern/libdiffpy gitlink. Keep this in sync when
# updating the submodule; source distributions must also build without Git.
set(DIFFPY_VERSION_MAJOR 1)
set(DIFFPY_VERSION_MINOR 4)
set(DIFFPY_VERSION_MICRO 0)
set(DIFFPY_VERSION_PATCH 0)
set(DIFFPY_VERSION 1004000000LL)
set(DIFFPY_VERSION_STR "1.4.0")
set(DIFFPY_VERSION_DATE "2026-06-24 00:40:51 -0600")
set(DIFFPY_GIT_SHA "b191c66df059aa1e4f96260b9df483454f32e084")

include("${CMAKE_CURRENT_LIST_DIR}/VerifyLibdiffpyRevision.cmake")
verify_libdiffpy_revision("${LIBDIFFPY_SOURCE_DIR}" "${DIFFPY_GIT_SHA}")
