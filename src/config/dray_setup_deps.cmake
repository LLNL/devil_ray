include(CMakeFindDependencyMacro)

###############################################################################
# Setup Conduit
###############################################################################
# If ZZZ_DIR not set, use known install path for Conduit and VTK-h
if(NOT CONDUIT_DIR)
  set(CONDUIT_DIR ${DRAY_CONDUIT_DIR})
endif()

###############################################################################
# Check for CONDUIT_DIR
###############################################################################
if(NOT CONDUIT_DIR)
    MESSAGE(FATAL_ERROR "Could not find Conduit. Conduit requires explicit CONDUIT_DIR.")
endif()

if(NOT EXISTS ${CONDUIT_DIR}/lib/cmake/conduit.cmake)
    MESSAGE(FATAL_ERROR "Could not find Conduit CMake include file (${CONDUIT_DIR}/lib/cmake/conduit.cmake)")
endif()

###############################################################################
# Import Conduit's CMake targets
###############################################################################
find_dependency(Conduit REQUIRED
               NO_DEFAULT_PATH
               PATHS ${CONDUIT_DIR}/lib/cmake)
#



if(NOT HDF5_DIR)
  set(HDF5_DIR ${DRAY_HDF5_DIR})
endif()

###############################################################################
# Check for CONDUIT_DIR
###############################################################################
if(NOT HDF5_DIR)
  MESSAGE(FATAL_ERROR "Could not find HDF5. HDF5 requires explicit HDF5_DIR.")
endif()

if(NOT EXISTS ${HDF5_DIR}/share/cmake/hdf5/hdf5-config.cmake)
  MESSAGE(FATAL_ERROR "Could not find hdf5 CMake include file (${HDF5_DIR}/share/cmake/hdf5/hdf5-config.cmake)")
endif()
###############################################################################
# Import Conduit's CMake targets
###############################################################################
find_dependency(hdf5 REQUIRED
               NO_DEFAULT_PATH
               PATHS ${HDF5_DIR}/share/cmake/hdf5/)





if(NOT RAJA_DIR)
  set(RAJA_DIR ${DRAY_RAJA_DIR})
endif()

###############################################################################
# Check for CONDUIT_DIR
###############################################################################
if(NOT RAJA_DIR)
  MESSAGE(FATAL_ERROR "Could not find RAJA. RAJA requires explicit RAJA_DIR.")
endif()

if(NOT EXISTS ${RAJA_DIR}/raja-config.cmake)
  MESSAGE(FATAL_ERROR "Could not find raja CMake include file (${RAJA_DIR}/raja-config.cmake)")
endif()
###############################################################################
# Import Conduit's CMake targets
###############################################################################
find_dependency(RAJA REQUIRED
               NO_DEFAULT_PATH
               PATHS ${RAJA_DIR})



if(NOT UMPRE_DIR)
  set(UMPIRE_DIR ${DRAY_UMPIRE_DIR})
endif()

###############################################################################
# Check for CONDUIT_DIR
###############################################################################
if(NOT UMPIRE_DIR)
  MESSAGE(FATAL_ERROR "Could not find Umpire. Umpire requires explicit UMPIRE_DIR.")
endif()

if(NOT EXISTS ${UMPIRE_DIR}/share/umpire/cmake/umpire-config.cmake)
  MESSAGE(FATAL_ERROR "Could not find umpire CMake include file (${UMPIRE_DIR}/share/umpire/cmake/umpire-config.cmake)")
endif()
###############################################################################
# Import Conduit's CMake targets
###############################################################################
find_dependency(umpire REQUIRED
               NO_DEFAULT_PATH
               PATHS ${UMPIRE_DIR})




if(NOT MFEM_DIR)
  set(MFEM_DIR ${DRAY_MFEM_DIR})
endif()

###############################################################################
# Check for CONDUIT_DIR
###############################################################################
if(NOT MFEM_DIR)
  MESSAGE(FATAL_ERROR "Could not find mfem. mfem requires explicit MFEM_DIR.")
endif()

if(NOT EXISTS ${MFEM_DIR}/MFEMConfig.cmake)
  MESSAGE(FATAL_ERROR "Could not find mfem CMake include file (${MFEM_DIR}/MFEMConfig.cmake)")
endif()
###############################################################################
# Import Conduit's CMake targets
###############################################################################
find_dependency(mfem REQUIRED
               NO_DEFAULT_PATH
               PATHS ${MFEM_DIR})

