set(dray_headers dray.hpp
                 array.hpp
                 array_internals.hpp
                 array_internals_base.hpp
                 array_registry.hpp
                 test.hpp
                 triangle_mesh.hpp
                 io/obj_reader.hpp
)
set(dray_sources dray.cpp
                 array.cpp
                 array_internals.cpp
                 array_internals_base.cpp
                 array_registry.cpp
                 triangle_mesh.cpp
                 test.cpp
)

set(dray_thirdparty_libs RAJA umpire)

if(ENABLE_OPENMP)
  list(APPEND dray_thirdparty_libs openmp)
endif()


blt_add_library(
  NAME dray
  SOURCES ${dray_sources}
  HEADERS ${dray_headers}
  DEPENDS_ON ${dray_thirdparty_libs}
)

target_include_directories(dray PRIVATE ../${CMAKE_CURRENT_SOURCE_DIR} )

install(TARGETS dray
        EXPORT dray
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION lib
)

install(FILES ${dray_headers} DESTINATION include/dray)
