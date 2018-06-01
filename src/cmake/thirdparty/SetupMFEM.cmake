if (NOT MFEM_DIR)
  message(FATAL_ERROR "ENABLE_MFEM=ON requires explicit 'MFEM_DIR'")
endif()

message(STATUS "Looking for MFEM in: ${MFEM_DIR}")

set(MFEM_DIR ${MFEM_DIR}/lib/cmake/mfem)
find_package(MFEM REQUIRED)
message(STATUS "Found MFEM")

blt_register_library(NAME mfem
                     INCLUDES ${MFEM_INCLUDE_DIRS}
                     LIBRARIES ${MFEM_LIBRARIES})

