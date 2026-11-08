cmake_minimum_required(VERSION 3.18)
project(flexscanx_core LANGUAGES CXX)

# The core uses std::thread, std::mutex and std::atomic (scheduler.hpp,
# scan_*.hpp, static_precompute.cpp, bindings_scan_execution.cpp).  The
# threading library must be linked explicitly: without it the conda-forge
# toolchain fails to link the test executable (undefined pthread_create /
# pthread_join) and builds the extension with unversioned pthread references
# that older glibc cannot resolve (FXA-160).
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# Option to build the Python extension module (ON by default for packaging)
option(FLEXSCANX_BUILD_PYEXT "Build Python extension module" ON)

if (FLEXSCANX_BUILD_PYEXT)
  # Extract version from pyproject.toml for single source of truth
  file(READ "${CMAKE_CURRENT_SOURCE_DIR}/pyproject.toml" PYPROJECT_CONTENT)
  string(REGEX MATCH "version = \"([0-9.]+)\"" _ "${PYPROJECT_CONTENT}")
  set(FLEXSCANX_VERSION ${CMAKE_MATCH_1})
  if(NOT FLEXSCANX_VERSION)
    message(WARNING "Could not extract version from pyproject.toml, using fallback")
    set(FLEXSCANX_VERSION "0.0.1")
  endif()
  message(STATUS "flexscanx version: ${FLEXSCANX_VERSION}")

  # Build-time digest of the native sources (FXA-168).  The editable install does
  # not rebuild the extension when only C++ changes, so a green Python suite can
  # run against a stale binary.  The digest is the SHA-256 of the sorted list of
  # "<relative path>\n<file sha256>\n" records over every maintained native
  # source plus this file; tests/test_fix20260907_native.py recomputes it and
  # fails loudly when the loaded extension does not match the tree.  Listing the
  # sources in CMAKE_CONFIGURE_DEPENDS re-runs this configure step whenever one
  # of them changes, so a rebuilt binary always carries the current digest.  (A
  # file added without re-configuring is caught by the same test, since Python
  # recomputes the digest over the tree; the glob is therefore not
  # CONFIGURE_DEPENDS, which would add a permanent glob-verification build step.)
  file(GLOB_RECURSE FLEXSCANX_NATIVE_SOURCES
       RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
       "${CMAKE_CURRENT_SOURCE_DIR}/src/flexscanx_core/*.cpp"
       "${CMAKE_CURRENT_SOURCE_DIR}/src/flexscanx_core/*.hpp"
       "${CMAKE_CURRENT_SOURCE_DIR}/src/flexscanx_core/*.in")
  list(APPEND FLEXSCANX_NATIVE_SOURCES "CMakeLists.txt")
  list(SORT FLEXSCANX_NATIVE_SOURCES)
  set(FLEXSCANX_SOURCE_MANIFEST "")
  foreach(_src IN LISTS FLEXSCANX_NATIVE_SOURCES)
    file(SHA256 "${CMAKE_CURRENT_SOURCE_DIR}/${_src}" _src_hash)
    string(APPEND FLEXSCANX_SOURCE_MANIFEST "${_src}\n${_src_hash}\n")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${_src}")
  endforeach()
  string(SHA256 FLEXSCANX_SOURCE_DIGEST "${FLEXSCANX_SOURCE_MANIFEST}")
  message(STATUS "flexscanx native source digest: ${FLEXSCANX_SOURCE_DIGEST}")

  # Generate version header
  configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/src/flexscanx_core/version.hpp.in"
    "${CMAKE_CURRENT_BINARY_DIR}/generated/version.hpp"
    @ONLY
  )

  # Find Python and pybind11 via scikit-build-core environment
  find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
  find_package(pybind11 CONFIG REQUIRED)

  # Sources
  # Note: middle_p_*.cpp and llr_*.cpp are header-only; their stub .cpp files
  # (which contained only a single #include) have been removed. The implementations
  # are compiled via explicit_inst.cpp which handles all template instantiations.
  set(SRC
      src/flexscanx_core/bindings.cpp
      src/flexscanx_core/bindings_support.cpp
      src/flexscanx_core/bindings_static.cpp
      src/flexscanx_core/bindings_scan.cpp
      src/flexscanx_core/bindings_scan_execution.cpp
      src/flexscanx_core/bindings_scan_result.cpp
      src/flexscanx_core/static_precompute.cpp
      src/flexscanx_core/scanner_flexible_poisson.cpp
      src/flexscanx_core/scanner_circular_poisson.cpp
      src/flexscanx_core/scanner_flexible_binomial.cpp
      src/flexscanx_core/scanner_circular_binomial.cpp
      src/flexscanx_core/algorithms/explicit_inst.cpp
  )

  pybind11_add_module(_core ${SRC})

  # Ensure C++17 and warnings reasonable
  target_compile_features(_core PRIVATE cxx_std_17)

  if (MSVC)
    target_compile_options(_core PRIVATE /EHsc /Zc:__cplusplus /O2 /DNDEBUG)
  else()
    # Use portable optimization flags for wheels (avoid -march=native). Avoid -ffast-math for numerical correctness.
    target_compile_options(_core PRIVATE -O3 -DNDEBUG)
  endif()

  # Enable IPO/LTO in Release and RelWithDebInfo builds (commonly used for profiling)
  if (CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    include(CheckIPOSupported)
    check_ipo_supported(RESULT ipo_supported OUTPUT ipo_msg)
    if (ipo_supported)
      set_property(TARGET _core PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
    endif()
  endif()

  # Include headers
  target_include_directories(_core PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_BINARY_DIR}/generated
  )
  target_link_libraries(_core PRIVATE Threads::Threads)

  # Install the extension into the Python package directory
  install(TARGETS _core
          LIBRARY DESTINATION flexscanx
          ARCHIVE DESTINATION flexscanx
          RUNTIME DESTINATION flexscanx)
endif()

# -----------------------------------------------------------------------------
# Tests: small C++ executable for core unit tests
# This target is for development/testing only, is NOT installed with the Python
# package, and is gated behind an option so packaging builds (pip install)
# never depend on it. Enable with:
#   pip install -e . --config-settings=cmake.define.FLEXSCANX_BUILD_CPP_TESTS=ON
# or -DFLEXSCANX_BUILD_CPP_TESTS=ON in a direct CMake configure.
# -----------------------------------------------------------------------------
option(FLEXSCANX_BUILD_CPP_TESTS "Build the flexscanx_core_tests unit-test executable" OFF)

if (FLEXSCANX_BUILD_CPP_TESTS)
  add_executable(flexscanx_core_tests
      tests/cpp/test_core.cpp
      src/flexscanx_core/static_precompute.cpp
      src/flexscanx_core/scanner_circular_poisson.cpp
      src/flexscanx_core/scanner_circular_binomial.cpp
      src/flexscanx_core/scanner_flexible_poisson.cpp
      src/flexscanx_core/scanner_flexible_binomial.cpp
  )

  # Include the src dir so tests can include headers like flexscanx_core/...
  target_include_directories(flexscanx_core_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_link_libraries(flexscanx_core_tests PRIVATE Threads::Threads)

  # Use same C++ standard and similar flags for tests
  set_property(TARGET flexscanx_core_tests PROPERTY CXX_STANDARD 17)
  if (MSVC)
    target_compile_options(flexscanx_core_tests PRIVATE /EHsc /Zc:__cplusplus /O2)
  else()
    target_compile_options(flexscanx_core_tests PRIVATE -O3)
  endif()
endif()

