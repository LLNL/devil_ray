###############################################################################
# export everything for a cmake build to be able to import with find_package
###############################################################################

if (NOT DEFINED DRAY_INSTALL_INCLUDE_DIR)
  set(DRAY_INSTALL_INCLUDE_DIR "include")
endif()

if (NOT DEFINED DRAY_INSTALL_CONFIG_DIR)
  set(DRAY_INSTALL_CONFIG_DIR "lib")
endif()

if (NOT DEFINED DRAY_INSTALL_LIB_DIR)
  set(DRAY_INSTALL_LIB_DIR "lib")
endif()

if (NOT DEFINED DRAY_INSTALL_BIN_DIR)
  set(DRAY_INSTALL_BIN_DIR "bin")
endif()

if (NOT DEFINED DRAY_INSTALL_SHARED_RESOURCES_DIR)
  set(DRAY_INSTALL_SHARED_RESOURCES_DIR "share/dray")
endif()

if (NOT DEFINED DRAY_INSTALL_CMAKE_MODULE_DIR)
  set(DRAY_INSTALL_CMAKE_MODULE_DIR "${DRAY_INSTALL_CONFIG_DIR}/cmake")
endif()

include(CMakePackageConfigHelpers)

# write version heler
write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/DRayConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)


# write cmake package config file
configure_package_config_file(
  DRayConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/DRayConfig.cmake
  INSTALL_DESTINATION ${DRAY_INSTALL_CONFIG_DIR}
  PATH_VARS
    DRAY_INSTALL_INCLUDE_DIR
    DRAY_INSTALL_LIB_DIR
    DRAY_INSTALL_BIN_DIR
    DRAY_INSTALL_SHARED_RESOURCES_DIR
    DRAY_INSTALL_CMAKE_MODULE_DIR
    )


# install everything needed
install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/DRayConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/DRayConfigVersion.cmake
    dray_setup_deps.cmake
    dray_setup_targets.cmake
    DESTINATION ${DRAY_INSTALL_CMAKE_MODULE_DIR})

  # bad things were being set in the cache
set(DRAY_RAJA_DIR  ${RAJA_DIR})
string(REPLACE "/share/raja/cmake" "" DRAY_RAJA_DIR ${DRAY_RAJA_DIR})
# generated the config.mk
configure_file ("${CMAKE_CURRENT_SOURCE_DIR}/dray_config.mk.in"
                "${CMAKE_CURRENT_BINARY_DIR}/dray_config.mk")

# install the config.mk
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/dray_config.mk" DESTINATION share/dray)
