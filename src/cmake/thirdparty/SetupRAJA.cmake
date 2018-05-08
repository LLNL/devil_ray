message(STATUS "Looking for RAJA in: ${RAJA_DIR}")
if (NOT RAJA_DIR)
  message(FATAL_ERROR "Must specify 'RAJA_DIR'")
endif()

set(RAJA_DIR ${RAJA_DIR}/share/raja/cmake)
find_package(RAJA REQUIRED)
message(STATUS "Found RAJA")
