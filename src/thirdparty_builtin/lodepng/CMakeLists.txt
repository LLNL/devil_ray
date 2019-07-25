################################
# Simple CMake setup for lodepng
################################

#
# lodepng sources
#

set(lodepng_sources
    lodepng.cpp
    )

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# we always build lodepng as lib
add_library(lodepng SHARED ${lodepng_sources} )

# install target for devil_ray lib
install(TARGETS lodepng
        EXPORT  dray
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION lib
)

#export(TARGETS lodepng FILE drayTargets.cmake)
