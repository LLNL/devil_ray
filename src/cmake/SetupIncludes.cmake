################################
#  Project Wide Includes
################################

# add lodepng include dir
include_directories(${PROJECT_SOURCE_DIR})

include_directories(${UMPIRE_INCLUDE_DIRS})

# add include dirs so units tests have access to the headers across
# libs and in unit tests
