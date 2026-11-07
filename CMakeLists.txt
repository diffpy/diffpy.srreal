cmake_minimum_required(VERSION 3.22)
project(diffpy_srreal LANGUAGES NONE)

# Keep this aligned with the pinned libdiffpy's CMakeLists.txt.
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(LIBDIFFPY_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/extern/libdiffpy")
if(NOT EXISTS "${LIBDIFFPY_SOURCE_DIR}/CMakeLists.txt")
    message(FATAL_ERROR
        "Missing libdiffpy sources. Initialize the submodule with "
        "'git submodule update --init --recursive', or use a source distribution "
        "that includes extern/libdiffpy."
    )
endif()

# libdiffpy resolves its version from git unless src/diffpy/version.hpp exists,
# as in its own source distribution. The srreal sdist has no git, so write that
# header here and ship it.
set(LIBDIFFPY_SDIST_VERSION_HEADER "${LIBDIFFPY_SOURCE_DIR}/src/diffpy/version.hpp")
if(SKBUILD_STATE STREQUAL "sdist")
    # Refresh git checkouts; preserve the header when repacking an sdist.
    if(EXISTS "${LIBDIFFPY_SOURCE_DIR}/.git" OR
       NOT EXISTS "${LIBDIFFPY_SDIST_VERSION_HEADER}")
        include("${LIBDIFFPY_SOURCE_DIR}/cmake/LibdiffpyVersion.cmake")
        libdiffpy_resolve_version()
        libdiffpy_write_version_header(
            "${LIBDIFFPY_SOURCE_DIR}/src/diffpy/version.tpl"
            "${LIBDIFFPY_SDIST_VERSION_HEADER}")
    endif()
    return()
endif()
# A header left over from a local sdist build would freeze the git checkout's
# version, so remove it.
if(EXISTS "${LIBDIFFPY_SOURCE_DIR}/.git" AND EXISTS "${LIBDIFFPY_SDIST_VERSION_HEADER}")
    message(STATUS "Removing stale ${LIBDIFFPY_SDIST_VERSION_HEADER}")
    file(REMOVE "${LIBDIFFPY_SDIST_VERSION_HEADER}")
endif()
enable_language(CXX)

# Conda is optional; system packages and CMAKE_PREFIX_PATH work as well.
if(DEFINED ENV{CONDA_PREFIX})
    list(APPEND CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}" "$ENV{CONDA_PREFIX}/Library")
endif()

find_package(Python 3.12 REQUIRED COMPONENTS Interpreter Development.Module)
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_VARIABLE nanobind_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY
)
find_package(nanobind 3.1...<4 CONFIG REQUIRED)

# PyPI wheels link Boost and GSL statically so they install without conda or
# system libraries. Conda and system builds keep the default shared linking.
option(SRREAL_STATIC_DEPS "Link Boost.Serialization and GSL statically" OFF)
set(SRREAL_DEPS_LICENSE_DIR "" CACHE PATH
    "Third-party license files to ship in the wheel with SRREAL_STATIC_DEPS")

if(SRREAL_STATIC_DEPS)
    set(Boost_USE_STATIC_LIBS ON)
endif()
find_package(Boost CONFIG REQUIRED COMPONENTS serialization)

if(SRREAL_STATIC_DEPS AND NOT WIN32)
    # FindGSL has no static switch; restrict the search to static archives.
    set(_srreal_saved_suffixes "${CMAKE_FIND_LIBRARY_SUFFIXES}")
    set(CMAKE_FIND_LIBRARY_SUFFIXES ".a")
endif()
find_package(GSL REQUIRED)
if(DEFINED _srreal_saved_suffixes)
    set(CMAKE_FIND_LIBRARY_SUFFIXES "${_srreal_saved_suffixes}")
    get_target_property(_srreal_boost_lib Boost::serialization LOCATION)
    foreach(_srreal_lib IN ITEMS "${_srreal_boost_lib}" "${GSL_LIBRARY}" "${GSL_CBLAS_LIBRARY}")
        if(NOT _srreal_lib MATCHES "\\.a$")
            message(FATAL_ERROR
                "SRREAL_STATIC_DEPS requires static libraries, found ${_srreal_lib}")
        endif()
    endforeach()
endif()

# Embed libdiffpy as an OBJECT library: every object is linked, so its static
# factory/serialization registrars are kept.
set(LIBDIFFPY_BUILD_OBJECT_LIB ON)
set(LIBDIFFPY_INSTALL OFF)
# ObjCryst conversion uses the public Python API, so it does not depend on
# pyobjcryst's private static library, compiler, REAL precision, or nanobind ABI.
set(LIBDIFFPY_ENABLE_OBJCRYST OFF)
set(LIBDIFFPY_BUILD_TESTS OFF)
# Runtime data installs next to the extension, in diffpy/srreal/data.
set(LIBDIFFPY_RUNTIME_RELPATH data)
add_subdirectory("${LIBDIFFPY_SOURCE_DIR}" libdiffpy EXCLUDE_FROM_ALL)
set_target_properties(diffpy PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)

file(GLOB SRREAL_SOURCES CONFIGURE_DEPENDS "src/extensions/*.cpp")
# Retain CPython-specific wheels; stable-ABI packaging is a separate change.
nanobind_add_module(srreal_ext NB_STATIC ${SRREAL_SOURCES})
target_link_libraries(srreal_ext PRIVATE libdiffpy::diffpy)
if(WIN32)
    target_compile_definitions(srreal_ext PRIVATE NOMINMAX)
endif()
set_target_properties(srreal_ext PROPERTIES INSTALL_RPATH_USE_LINK_PATH ON)
install(TARGETS srreal_ext LIBRARY DESTINATION diffpy/srreal)
install(DIRECTORY "${LIBDIFFPY_SOURCE_DIR}/src/runtime/"
        DESTINATION diffpy/srreal/data
        FILES_MATCHING PATTERN "*.dat" PATTERN "bvparm2011sel.cif")
install(FILES "${LIBDIFFPY_SOURCE_DIR}/LICENSE.rst"
              "${LIBDIFFPY_SOURCE_DIR}/LICENSE_DANSE.rst"
        DESTINATION diffpy/srreal/licenses/libdiffpy)
# Statically linked Boost and GSL code ships inside the extension, so their
# licenses ship with it.
if(SRREAL_STATIC_DEPS AND SRREAL_DEPS_LICENSE_DIR)
    install(DIRECTORY "${SRREAL_DEPS_LICENSE_DIR}/"
            DESTINATION diffpy/srreal/licenses)
endif()
