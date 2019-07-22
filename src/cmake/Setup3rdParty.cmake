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
include(cmake/thirdparty/SetupHDF5.cmake)

################################
# Setup Python
################################
include(cmake/thirdparty/SetupPython.cmake)

################################
# Setup MFEM
################################
if(ENABLE_MFEM)
  include(cmake/thirdparty/SetupMFEM.cmake)
endif()
