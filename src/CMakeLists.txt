cmake_minimum_required(VERSION 3.2)

################################
# Build Options
################################
option(BUILD_SHARED_LIBS  "Build shared libraries"    ON)
option(ENABLE_TESTS       "Build tests"               ON)
option(ENABLE_MPI         "Build MPI Support"         ON)

# Set BLT Options
################################
# forward the build shared libs option to blt
set(ENABLE_SHARED_LIBS ${BUILD_SHARED_LIBS} CACHE INTERNAL "")

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
# Setup 3rd Party Libs
################################
include(cmake/Setup3rdParty.cmake)

add_subdirectory(rtracer)
