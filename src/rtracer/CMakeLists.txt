set(rtracer_headers rtracer.hpp
                    array_internals.hpp
                    array_internals_base.hpp
                    array_registry.hpp
                    test.hpp
)
set(rtracer_sources rtracer.cpp
                    array_internals.cpp
                    array_internals_base.cpp
                    array_registry.cpp
                    test.cpp
)
set(rtracer_thridparty_libs RAJA umpire)


blt_add_library(
  NAME rtracer
  SOURCES ${rtracer_sources}
  HEADERS ${rtracer_headers}
  DEPENDS_ON ${rtracer_thridparty_libs}
)

target_include_directories(rtracer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} )

install(TARGETS rtracer
        EXPORT rtracer
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION lib
)

install(FILES ${rtracer_headers} DESTINATION include/rtracer)
