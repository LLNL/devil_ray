################################
# devil ray 3rd Party Dependencies
################################

###############################################################################
# gtest, fruit, mpi, cuda, openmp, sphinx and doxygen are handled by blt
###############################################################################

################################################################
################################################################
#
# 3rd Party Libs
#
################################################################
################################################################

include(CMakeFindDependencyMacro)

################################
# Setup RAJA
################################
include(cmake/thirdparty/SetupRAJA.cmake)

################################
# Setup Umpire
################################
include(cmake/thirdparty/SetupUmpire.cmake)

################################
# Setup Conduit
################################
include(cmake/thirdparty/SetupConduit.cmake)

################################
# Setup Conduit
################################
if(HDF5_DIR)
  include(cmake/thirdparty/SetupHDF5.cmake)
endif()

################################
# Setup MFEM
################################
include(cmake/thirdparty/SetupMFEM.cmake)

################################
# Setup APComp
################################
include(cmake/thirdparty/SetupAPComp.cmake)

# used to track the blt imported libs we need to export
set(DRAY_BLT_TPL_DEPS_EXPORTS)

##################################
# Export BLT Targets when needed
##################################

blt_list_append(TO DRAY_BLT_TPL_DEPS_EXPORTS ELEMENTS cuda cuda_runtime IF ENABLE_CUDA)
# OFF FOR HIP EXPERIMENT
#blt_list_append(TO DRAY_BLT_TPL_DEPS_EXPORTS ELEMENTS hip hip_runtime IF ENABLE_HIP)
blt_list_append(TO DRAY_BLT_TPL_DEPS_EXPORTS ELEMENTS openmp IF ENABLE_OPENMP)

# cmake < 3.15, we use BLT's mpi target and need to export
# it for use downstream
if( ${CMAKE_VERSION} VERSION_LESS "3.15.0" )
    blt_list_append(TO DRAY_BLT_TPL_DEPS_EXPORTS ELEMENTS mpi IF ENABLE_MPI)
endif()

foreach(dep ${DRAY_BLT_TPL_DEPS_EXPORTS})
    # If the target is EXPORTABLE, add it to the export set
    get_target_property(_is_imported ${dep} IMPORTED)
    if(NOT ${_is_imported})
        install(TARGETS              ${dep}
                EXPORT               dray
                DESTINATION          lib)
        # Namespace target to avoid conflicts
        set_target_properties(${dep} PROPERTIES EXPORT_NAME dray::blt_${dep})
    endif()
endforeach()
