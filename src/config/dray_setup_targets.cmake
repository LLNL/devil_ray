set(DRAY_INCLUDE_DIRS "${DRAY_INSTALL_PREFIX}/include")

#
# Probe Ascent Features
#

# create convenience target that bundles all reg ascent deps (ascent::ascent)

add_library(dray::dray INTERFACE IMPORTED)

set_property(TARGET dray::dray
             APPEND PROPERTY
             INTERFACE_INCLUDE_DIRECTORIES "${DRAY_INSTALL_PREFIX}/include/")

set_property(TARGET dray::dray
             PROPERTY INTERFACE_LINK_LIBRARIES
             dray)

# try to include conduit with new exports
#if(TARGET conduit::conduit)
#    set_property(TARGET ascent::ascent
#                 APPEND PROPERTY INTERFACE_LINK_LIBRARIES
#                 conduit::conduit)
#else()
#    # if not, bottle conduit
#    set_property(TARGET ascent::ascent
#                 APPEND PROPERTY
#                 INTERFACE_INCLUDE_DIRECTORIES ${CONDUIT_INCLUDE_DIRS})
#
#    set_property(TARGET ascent::ascent
#                 APPEND PROPERTY INTERFACE_LINK_LIBRARIES
#                 conduit conduit_relay conduit_blueprint)
#endif()
#
#if(VTKH_FOUND)
#    set_property(TARGET ascent::ascent
#                 APPEND PROPERTY INTERFACE_LINK_LIBRARIES
#                 vtkh)
#endif()

#if(NOT DRay_FIND_QUIETLY)
#
#    message(STATUS "ASCENT_VERSION             = ${ASCENT_VERSION}")
#    message(STATUS "ASCENT_INSTALL_PREFIX      = ${ASCENT_INSTALL_PREFIX}")
#    message(STATUS "ASCENT_INCLUDE_DIRS        = ${ASCENT_INCLUDE_DIRS}")
#    message(STATUS "ASCENT_FORTRAN_ENABLED     = ${ASCENT_FORTRAN_ENABLED}")
#    message(STATUS "ASCENT_PYTHON_EXECUTABLE   = ${ASCENT_PYTHON_EXECUTABLE}")
#
#    set(_print_targets "ascent::ascent")
#    if(ASCENT_MPI_ENABLED)
#        set(_print_targets "${_print_targets} ascent::ascent_mpi")
#    endif()
#
#    message(STATUS "Ascent imported targets: ${_print_targets}")
#    unset(_print_targets)
#
#endif()
