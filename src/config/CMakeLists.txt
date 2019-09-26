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
