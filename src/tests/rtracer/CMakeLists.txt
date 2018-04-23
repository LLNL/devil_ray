set(BASIC_TESTS t_rtracer_smoke
                t_rtracer_test
)

message(STATUS "Adding basic rtracer unit tests")
foreach(TEST ${BASIC_TESTS})
  message(STATUS "adding test ${TEST}")
  add_cpp_test(TEST ${TEST} DEPENDS_ON rtracer)
endforeach()
