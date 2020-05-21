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
