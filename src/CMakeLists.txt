cmake_minimum_required(VERSION 3.2)

project(devil_ray)

################################
# Build Options
################################
option(BUILD_SHARED_LIBS  "Build shared libraries"    ON)
option(ENABLE_TESTS       "Build tests"               ON)
option(ENABLE_OPENMP      "Build OpenMP Support"      ON)
option(ENABLE_CUDA        "Build CUDA Support"        OFF)
option(ENABLE_MFEM        "Build MFEM Support"        OFF)
option(DEBUG_CPU_ONLY     "Enable CPU debugging"      OFF)
option(ENABLE_LOGGING     "Enable logging"            ON)
option(ENABLE_STATS       "Enable stats"              ON)

# Set BLT Options
################################
if(NOT BLT_SOURCE_DIR)
      set(BLT_SOURCE_DIR "blt")
endif()

if (NOT BLT_CXX_STD)
    set(BLT_CXX_STD "c++11" CACHE STRING "")
endif()

# don't use BLT's all warnings feature
set(ENABLE_ALL_WARNINGS OFF CACHE INTERNAL "")

################################
# Init BLT
################################
include(blt/SetupBLT.cmake)

################################
# Basic CMake Setup
################################
include(cmake/CMakeBasics.cmake)

################################
# Basic CMake Setup
################################
include(blt/cmake/SetupCompilerOptions.cmake)

################################
# Setup 3rd Party Libs
################################
include(cmake/Setup3rdParty.cmake)

################################
# Setup test helpers
################################
include(cmake/SetupTests.cmake)

################################
# Setup project includes
################################
include(cmake/SetupIncludes.cmake)

add_subdirectory(dray)

add_subdirectory(thirdparty_builtin)

add_subdirectory(tests)
