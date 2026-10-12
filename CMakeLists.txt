cmake_minimum_required(VERSION 3.20)
project(diffpy_srreal LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

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

find_path(LIBDIFFPY_INCLUDE_DIR NAMES diffpy/version.hpp REQUIRED)
find_library(LIBDIFFPY_LIBRARY NAMES diffpy libdiffpy REQUIRED)
if(UNIX AND NOT APPLE)
    add_library(LibDiffpy::LibDiffpy SHARED IMPORTED)
    # SCons builds may omit SONAME; link by name instead of an absolute path.
    set_target_properties(LibDiffpy::LibDiffpy PROPERTIES IMPORTED_NO_SONAME TRUE)
else()
    add_library(LibDiffpy::LibDiffpy UNKNOWN IMPORTED)
endif()
set_target_properties(LibDiffpy::LibDiffpy PROPERTIES
    IMPORTED_LOCATION "${LIBDIFFPY_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${LIBDIFFPY_INCLUDE_DIR}"
)

file(GLOB SRREAL_SOURCES CONFIGURE_DEPENDS "src/extensions/*.cpp")
# Retain CPython-specific wheels; stable-ABI packaging is a separate change.
nanobind_add_module(srreal_ext NB_STATIC ${SRREAL_SOURCES})
target_link_libraries(srreal_ext PRIVATE
    LibDiffpy::LibDiffpy Boost::serialization GSL::gsl GSL::gslcblas)
if(WIN32)
    target_compile_definitions(srreal_ext PRIVATE _USE_MATH_DEFINES NOMINMAX)
endif()
set_target_properties(srreal_ext PROPERTIES INSTALL_RPATH_USE_LINK_PATH ON)
install(TARGETS srreal_ext LIBRARY DESTINATION diffpy/srreal)
