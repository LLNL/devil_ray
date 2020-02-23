message(STATUS "Looking for RAJA in: ${RAJA_DIR}")
if (NOT RAJA_DIR)
  message(FATAL_ERROR "Must specify 'RAJA_DIR'")
endif()

set(TEMP_RAJA_DIR ${RAJA_DIR})
find_dependency(RAJA REQUIRED
               NO_DEFAULT_PATH
               PATHS ${RAJA_DIR}/share/raja/cmake)

# prevent RAJA from setting the raja dir
set(RAJA_DIR ${TEMP_RAJA_DIR})
message(STATUS "Found RAJA")
