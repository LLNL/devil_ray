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



if(NOT RAJA_DIR)
  set(RAJA_DIR ${DRAY_RAJA_DIR})
endif()

###############################################################################
# Check for RAJA_DIR
###############################################################################

if(NOT RAJA_DIR)
  MESSAGE(FATAL_ERROR "Could not find RAJA. RAJA requires explicit RAJA_DIR.")
endif()

# We can't trust RAJA dir to be the base dir, so do a sanity check
string(REPLACE "/share/raja/cmake" "" FILTERED_RAJA_DIR ${RAJA_DIR})

if(NOT EXISTS ${FILTERED_RAJA_DIR}/share/raja/cmake/raja-config.cmake)
  MESSAGE(FATAL_ERROR "Could not find raja CMake include file (${FILTERED_RAJA_DIR}/share/raja/cmake/raja-config.cmake)")
endif()
###############################################################################
# Import Conduit's RAJA targets
###############################################################################
find_dependency(RAJA REQUIRED
               NO_DEFAULT_PATH
               PATHS ${FILTERED_RAJA_DIR}/share/raja/cmake)


if(NOT UMPIRE_DIR)
  set(UMPIRE_DIR ${DRAY_UMPIRE_DIR})
endif()

###############################################################################
# Check for UMPIRE_DIR
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


if(NOT APCOMP_DIR)
  set(APCOMP_DIR ${DRAY_APCOMP_DIR})
endif()

###############################################################################
# Check for UMPIRE_DIR
###############################################################################
if(NOT APCOMP_DIR)
  MESSAGE(FATAL_ERROR "Could not find APComp. Dray requires explicit APCOMP_DIR.")
endif()

set(APComp_DIR ${APCOMP_DIR}/lib/cmake/)
find_package(APComp REQUIRED)

if(NOT EXISTS ${APCOMP_DIR}/lib/cmake/APCompConfig.cmake)
  MESSAGE(FATAL_ERROR "Could not find apcomp CMake include file (${APCOMP_DIR}/lib/cmake/APCompConfig.cmake)")
endif()
###############################################################################
# Import Conduit's CMake targets
###############################################################################
find_package(APComp REQUIRED
             NO_DEFAULT_PATH
             PATHS ${APCOMP_DIR})

# MFEM is not cmake based so we will
# have to add the SetupMFEM logic here

#if(NOT MFEM_DIR)
#  set(MFEM_DIR ${DRAY_MFEM_DIR})
#endif()
#
################################################################################
## Check for CONDUIT_DIR
################################################################################
#if(NOT MFEM_DIR)
#  MESSAGE(FATAL_ERROR "Could not find mfem. mfem requires explicit MFEM_DIR.")
#endif()
#
#if(NOT EXISTS ${MFEM_DIR}/MFEMConfig.cmake)
#  MESSAGE(FATAL_ERROR "Could not find mfem CMake include file (${MFEM_DIR}/MFEMConfig.cmake)")
#endif()
################################################################################
## Import Conduit's CMake targets
################################################################################
#find_dependency(mfem REQUIRED
#               NO_DEFAULT_PATH
#               PATHS ${MFEM_DIR})
#
