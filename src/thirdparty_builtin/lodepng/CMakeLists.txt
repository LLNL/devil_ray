################################
# Simple CMake setup for lodepng
################################

#
# lodepng sources
#

set(lodepng_sources
    lodepng.cpp
    )

set(lodepng_headers
    lodepng.h
    )

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

blt_add_library(
  NAME dray_lodepng
  SOURCES ${lodepng_sources}
  HEADERS ${lodepng_headers}
)

# install target for devil_ray lib
install(TARGETS dray_lodepng
        EXPORT  dray
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION lib
)

#export(TARGETS lodepng FILE drayTargets.cmake)
