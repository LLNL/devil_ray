# avoid warnings for project commands w/o VERSION
cmake_policy(SET CMP0048 NEW)

project(devil_ray VERSION "0.0.1")

# Set default build types
include(cmake/DRayBuildType.cmake)

################################
# Build Options
################################
option(BUILD_SHARED_LIBS  "Build shared libraries"    OFF)
option(DRAY_ENABLE_TESTS  "Build tests"               ON)
option(DRAY_ENABLE_UTILS  "Build utilities"           ON)
option(ENABLE_OPENMP      "Build OpenMP Support"      ON)
option(ENABLE_MPI         "Build MPI Support"         ON)
option(ENABLE_CUDA        "Build CUDA Support"        ON)
option(ENABLE_HIP         "Build HIP Support"         ON)
option(ENABLE_MFEM        "Build MFEM Support"        ON)
option(ENABLE_DOCS        "Build documentation"       OFF)
option(DEBUG_CPU_ONLY     "Enable CPU debugging"      OFF)
option(ENABLE_LOGGING     "Enable logging"            ON)
option(ENABLE_STATS       "Enable stats"              ON)
option(DRAY_USE_DOUBLE_PRECISION "Use double precision" OFF)
option(ENABLE_SERIAL      "Build serial Support (no-mpi)" ON)

if(ENABLE_CUDA)
  cmake_minimum_required(VERSION 3.14)
else()
  cmake_minimum_required(VERSION 3.9)
endif()

################################
# Init BLT
################################
# This also includes 
# our BLT defaults
include(cmake/SetupBLT.cmake)

################################
# Basic CMake Setup
################################
include(cmake/CMakeBasics.cmake)

################################
# Basic CMake Setup
################################
include(${BLT_SOURCE_DIR}/cmake/SetupCompilerOptions.cmake)

################################
# Setup 3rd Party Libs
################################
include(cmake/Setup3rdParty.cmake)

################################
# Setup project includes
################################
include(cmake/SetupIncludes.cmake)

add_subdirectory(dray)

add_subdirectory(thirdparty_builtin)

if(DRAY_ENABLE_UTILS)
  message(status "Dray Utilities Enabled")
  add_subdirectory(utilities)
endif()

################################
# Setup test helpers and tests
################################
if(DRAY_ENABLE_TESTS)
  message(STATUS "Dray Tests Enabled")
  include(cmake/SetupTests.cmake)
  add_subdirectory(tests)
else()
  message(STATUS "Dray Tests Disabled")
endif()

add_subdirectory(config)

if(ENABLE_DOCS)
  add_subdirectory(docs)
endif()

install(EXPORT dray DESTINATION lib/cmake)
