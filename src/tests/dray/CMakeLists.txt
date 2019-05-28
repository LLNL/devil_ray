
set(DATA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/data/")
configure_file(test_config.h.in "${CMAKE_CURRENT_SOURCE_DIR}/test_config.h" @ONLY)

set(BASIC_TESTS t_dray_smoke
                t_dray_array
                t_dray_color_table
                t_dray_obj_reader
                t_dray_matrix
                t_dray_registry
                t_dray_tri_benchmark
                t_dray_test
                t_dray_vec
                t_dray_high_order_shape
                t_dray_newton_solve
                t_dray_cool_shapes
                t_dray_appstats
)

set(dray_test_deps dray openmp)

if(ENABLE_MFEM)
  list(APPEND dray_test_deps mfem)
endif()

message(STATUS "Adding basic dray unit tests")
foreach(TEST ${BASIC_TESTS})
  message(STATUS "adding test ${TEST}")
  add_cpp_test(TEST ${TEST} DEPENDS_ON ${dray_test_deps})
endforeach()

if(ENABLE_MFEM)
  set(MFEM_TESTS
    #t_dray_mfem_h1
    #t_dray_blueprint_reader
                 t_dray_mfem_reader
                 t_dray_tripple_point
                 t_dray_taylor_green
  )

  message(STATUS "Adding mfem dray unit tests")
  foreach(TEST ${MFEM_TESTS})
    message(STATUS "adding test ${TEST}")
    add_cpp_test(TEST ${TEST} DEPENDS_ON ${dray_test_deps})
  endforeach()
endif()
