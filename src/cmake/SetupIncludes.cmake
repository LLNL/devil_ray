################################
#  Project Wide Includes
################################

# add lodepng include dir
include_directories(${PROJECT_SOURCE_DIR})

include_directories(${UMPIRE_INCLUDE_DIRS})

if(ENABLE_MFEM)
  include_directories(${MFEM_INCLUDE_DIRS})
endif()

include_directories(${PROJECT_SOURCE_DIR}/thirdparty_builtin/lodepng)
include_directories(${PROJECT_SOURCE_DIR}/thirdparty_builtin/tiny_obj)

# add include dirs so units tests have access to the headers across
# libs and in unit tests
