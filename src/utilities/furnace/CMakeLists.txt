set(furnace_thirdparty_libs dray)

################################################
# openmp support
################################################
if(ENABLE_OPENMP)
  list(APPEND furnace_thirdparty_libs openmp)
endif()

################################################
# point location furnace
################################################
blt_add_executable(
  NAME point_location
  SOURCES point_location.cpp
  DEPENDS_ON ${furnace_thirdparty_libs}
  OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}
)

if(ENABLE_STATS)
  target_compile_definitions(point_location PRIVATE "DRAY_STATS")
endif()

configure_file(point_config.yaml ${CMAKE_CURRENT_BINARY_DIR}/point_config.yaml COPYONLY)

################################################
# surface intersection furnace
################################################
blt_add_executable(
  NAME surface_intersection
  SOURCES surface_intersection.cpp
  DEPENDS_ON ${furnace_thirdparty_libs}
  OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}
)

if(ENABLE_STATS)
  target_compile_definitions(surface_intersection PRIVATE "DRAY_STATS")
endif()

configure_file(intersection_config.yaml ${CMAKE_CURRENT_BINARY_DIR}/intersection_config.yaml COPYONLY)

################################################
# surface intersection furnace
################################################
blt_add_executable(
  NAME volume_rendering
  SOURCES volume_rendering.cpp
  DEPENDS_ON ${furnace_thirdparty_libs}
  OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}
)

if(ENABLE_STATS)
  target_compile_definitions(volume_rendering PRIVATE "DRAY_STATS")
endif()

#configure_file(point_config.yaml ${CMAKE_CURRENT_BINARY_DIR}/point_config.yaml COPYONLY)

################################################
# degree raising furnace
################################################
blt_add_executable(
  NAME degree_raising_accuracy
  SOURCES degree_raising_accuracy.cpp
  DEPENDS_ON ${furnace_thirdparty_libs}
  OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}
)

if(ENABLE_STATS)
  target_compile_definitions(degree_raising_accuracy PRIVATE "DRAY_STATS")
endif()

#configure_file(point_config.yaml ${CMAKE_CURRENT_BINARY_DIR}/point_config.yaml COPYONLY)

################################################
# degree raised location furnace
################################################
blt_add_executable(
  NAME degree_raised_location
  SOURCES degree_raised_location.cpp
  DEPENDS_ON ${furnace_thirdparty_libs}
  OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}
)

if(ENABLE_STATS)
  target_compile_definitions(degree_raised_location PRIVATE "DRAY_STATS")
endif()

#configure_file(point_config.yaml ${CMAKE_CURRENT_BINARY_DIR}/point_config.yaml COPYONLY)

install(FILES point_config.yaml intersection_config.yaml
        DESTINATION utilities/furnace
        )
