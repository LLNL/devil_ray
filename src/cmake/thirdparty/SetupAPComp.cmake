message(STATUS "Looking for AP Compositor in: ${APCOMP_DIR}")
if (NOT APCOMP_DIR)
  message(FATAL_ERROR "Must specify 'APCOMP_DIR'")
endif()

set(apcomp_DIR ${APCOMP_DIR}/lib/cmake/)
find_package(apcomp REQUIRED)
message(STATUS "Found AP Compositor")
