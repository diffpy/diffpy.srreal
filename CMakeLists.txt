cmake_minimum_required(VERSION 3.20)
project(diffpy_srreal LANGUAGES NONE)

# Keep this aligned with the pinned libdiffpy's src/SConscript.
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(LIBDIFFPY_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/extern/libdiffpy")
if(NOT EXISTS "${LIBDIFFPY_SOURCE_DIR}/src/diffpy/version.tpl")
    message(FATAL_ERROR
        "Missing libdiffpy sources. Initialize the submodule with "
        "'git submodule update --init --recursive', or use a source distribution "
        "that includes extern/libdiffpy."
    )
endif()

# Read version information from the submodule; sdists carry the generated header.
set(LIBDIFFPY_VERSION_HEADER "${CMAKE_CURRENT_BINARY_DIR}/include/diffpy/version.hpp")
set(LIBDIFFPY_SDIST_VERSION_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/_build/libdiffpy-version.hpp")
if(EXISTS "${LIBDIFFPY_SOURCE_DIR}/.git")
    file(STRINGS "${LIBDIFFPY_SOURCE_DIR}/site_scons/fallback_version.py"
         LIBDIFFPY_FALLBACK_VERSION REGEX "^FALLBACK_VERSION =")
    if(NOT LIBDIFFPY_FALLBACK_VERSION MATCHES "([0-9]+)\\.([0-9]+)\\.([0-9]+)")
        message(FATAL_ERROR "Cannot read the libdiffpy base version.")
    endif()
    set(DIFFPY_VERSION_MAJOR "${CMAKE_MATCH_1}")
    set(DIFFPY_VERSION_MINOR "${CMAKE_MATCH_2}")
    set(DIFFPY_VERSION_MICRO "${CMAKE_MATCH_3}")
    set(DIFFPY_VERSION_PATCH 0)
    set(DIFFPY_VERSION_STR "${DIFFPY_VERSION_MAJOR}.${DIFFPY_VERSION_MINOR}.${DIFFPY_VERSION_MICRO}")
    math(EXPR DIFFPY_VERSION "${DIFFPY_VERSION_MAJOR} * 1000000000 + ${DIFFPY_VERSION_MINOR} * 1000000 + ${DIFFPY_VERSION_MICRO} * 1000")
    set(DIFFPY_VERSION "${DIFFPY_VERSION}LL")
    find_package(Git REQUIRED)
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" -C "${LIBDIFFPY_SOURCE_DIR}" log -1 "--format=%H;%ci"
        OUTPUT_VARIABLE LIBDIFFPY_COMMIT_INFO
        OUTPUT_STRIP_TRAILING_WHITESPACE
        COMMAND_ERROR_IS_FATAL ANY
    )
    list(GET LIBDIFFPY_COMMIT_INFO 0 DIFFPY_GIT_SHA)
    list(GET LIBDIFFPY_COMMIT_INFO 1 DIFFPY_VERSION_DATE)
    configure_file("${LIBDIFFPY_SOURCE_DIR}/src/diffpy/version.tpl"
                   "${LIBDIFFPY_VERSION_HEADER}")
else()
    configure_file("${LIBDIFFPY_SDIST_VERSION_HEADER}"
                   "${LIBDIFFPY_VERSION_HEADER}" COPYONLY)
endif()
if(SKBUILD_STATE STREQUAL "sdist")
    configure_file("${LIBDIFFPY_VERSION_HEADER}"
                   "${LIBDIFFPY_SDIST_VERSION_HEADER}" COPYONLY)
    return()
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
find_package(Boost CONFIG REQUIRED COMPONENTS serialization)
find_package(GSL REQUIRED)

set(DIFFPY_HAS_OBJCRYST 0)
configure_file("${LIBDIFFPY_SOURCE_DIR}/src/diffpy/features.tpl"
               "include/diffpy/features.hpp")

file(GLOB LIBDIFFPY_SOURCES CONFIGURE_DEPENDS
    "${LIBDIFFPY_SOURCE_DIR}/src/diffpy/*.cpp"
    "${LIBDIFFPY_SOURCE_DIR}/src/diffpy/srreal/*.cpp"
)
# ObjCryst conversion uses the public Python API, so it does not depend on
# pyobjcryst's private static library, compiler, REAL precision, or nanobind ABI.
list(FILTER LIBDIFFPY_SOURCES EXCLUDE REGEX "ObjCryst")
# Include every object: libdiffpy has static factory/serialization registrars
# that an ordinary static archive link can otherwise discard.
add_library(diffpy_core OBJECT ${LIBDIFFPY_SOURCES})
set_target_properties(diffpy_core PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)
target_include_directories(diffpy_core PUBLIC
    "${CMAKE_CURRENT_BINARY_DIR}/include"
    "${LIBDIFFPY_SOURCE_DIR}/src"
)
target_compile_definitions(diffpy_core PRIVATE DIFFPYRUNTIMERELPATH=data)
if(WIN32)
    target_compile_definitions(diffpy_core PUBLIC _USE_MATH_DEFINES NOMINMAX)
endif()
target_link_libraries(diffpy_core PUBLIC Boost::serialization GSL::gsl GSL::gslcblas ${CMAKE_DL_LIBS})

file(GLOB SRREAL_SOURCES CONFIGURE_DEPENDS "src/extensions/*.cpp")
# Retain CPython-specific wheels; stable-ABI packaging is a separate change.
nanobind_add_module(srreal_ext NB_STATIC ${SRREAL_SOURCES})
target_link_libraries(srreal_ext PRIVATE diffpy_core)
set_target_properties(srreal_ext PROPERTIES INSTALL_RPATH_USE_LINK_PATH ON)
install(TARGETS srreal_ext LIBRARY DESTINATION diffpy/srreal)
install(DIRECTORY "${LIBDIFFPY_SOURCE_DIR}/src/runtime/"
        DESTINATION diffpy/srreal/data
        FILES_MATCHING PATTERN "*.dat" PATTERN "bvparm2011sel.cif")
install(FILES "${LIBDIFFPY_SOURCE_DIR}/LICENSE.rst"
              "${LIBDIFFPY_SOURCE_DIR}/LICENSE_DANSE.rst"
        DESTINATION diffpy/srreal/licenses/libdiffpy)
