cmake_minimum_required(VERSION 3.20)
project(diffpy_srreal LANGUAGES CXX)

# Keep this aligned with the pinned libdiffpy's src/SConscript.
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Conda is optional; system packages and CMAKE_PREFIX_PATH work as well.
if(DEFINED ENV{CONDA_PREFIX})
    list(APPEND CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}" "$ENV{CONDA_PREFIX}/Library")
endif()

find_package(Python 3.12 REQUIRED COMPONENTS Interpreter Development.Module NumPy)
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_VARIABLE nanobind_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY
)
find_package(nanobind CONFIG REQUIRED)
find_package(Boost CONFIG REQUIRED COMPONENTS serialization)
find_package(GSL REQUIRED)

set(LIBDIFFPY_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/extern/libdiffpy")
if(NOT EXISTS "${LIBDIFFPY_SOURCE_DIR}/src/diffpy/version.tpl")
    message(FATAL_ERROR "Missing libdiffpy sources. Run git submodule update --init --recursive.")
endif()

include(cmake/LibdiffpyVersion.cmake)
configure_file("${LIBDIFFPY_SOURCE_DIR}/src/diffpy/version.tpl"
               "include/diffpy/version.hpp")
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
target_link_libraries(diffpy_core PUBLIC Boost::serialization GSL::gsl GSL::gslcblas ${CMAKE_DL_LIBS})

file(GLOB SRREAL_SOURCES CONFIGURE_DEPENDS "src/extensions/*.cpp")
# Keep the CPython-specific ABI while the bindings use the NumPy C API and
# CPython macros. Do not label these wheels abi3 without porting those calls.
nanobind_add_module(srreal_ext NB_STATIC ${SRREAL_SOURCES})
target_link_libraries(srreal_ext PRIVATE diffpy_core Python::NumPy)
set_target_properties(srreal_ext PROPERTIES INSTALL_RPATH_USE_LINK_PATH ON)
install(TARGETS srreal_ext LIBRARY DESTINATION diffpy/srreal)
install(DIRECTORY "${LIBDIFFPY_SOURCE_DIR}/src/runtime/"
        DESTINATION diffpy/srreal/data
        FILES_MATCHING PATTERN "*.dat" PATTERN "bvparm2011sel.cif")
install(FILES "${LIBDIFFPY_SOURCE_DIR}/LICENSE.txt"
              "${LIBDIFFPY_SOURCE_DIR}/LICENSE_DANSE.txt"
        DESTINATION diffpy/srreal/licenses/libdiffpy)
