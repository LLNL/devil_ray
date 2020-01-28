
set(DATA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/data/")
configure_file(test_config.h.in "${CMAKE_CURRENT_SOURCE_DIR}/test_config.h" @ONLY)

set(BASIC_TESTS t_dray_smoke
                t_dray_array
                t_dray_color_table
                ##t_dray_ambient_occlusion
                ##t_dray_obj_reader
                #t_dray_matrix
                #t_dray_registry
                t_dray_slice
                t_dray_multi_render
                ##t_dray_tri_benchmark
                #t_dray_test
                #t_dray_vec
                #t_dray_high_order_shape
                #t_dray_newton_solve
                #t_dray_cool_shapes
                #t_dray_subdivision
                ##t_dray_performance_tests
                #t_dray_attractors
                ## t_dray_get_sub_bounds  ## has face elements
                t_dray_isosurface
                t_dray_scalar_renderer
                t_dray_volume_render
                t_dray_face_render
                #t_dray_sedov
                #t_dray_taylor_green
                ## t_dray_crazy_hex
                #t_dray_crazy_quad
                #t_dray_triangle
                #t_dray_element
)

set(dray_test_deps dray)

if(ENABLE_OPENMP)
  list(APPEND dray_test_deps openmp)
endif()

list(APPEND dray_test_deps mfem)

if(ENABLE_STATS)
  list(APPEND BASIC_TESTS t_dray_appstats)
endif()

message(STATUS "Adding basic dray unit tests")
foreach(TEST ${BASIC_TESTS})
  message(STATUS "adding test ${TEST}")
  add_cpp_test(TEST ${TEST} DEPENDS_ON ${dray_test_deps})
endforeach()

message(STATUS "Adding mfem dray unit tests")
foreach(TEST ${MFEM_TESTS})
  message(STATUS "adding test ${TEST}")
  add_cpp_test(TEST ${TEST} DEPENDS_ON ${dray_test_deps})
endforeach()


if(PYTHON_FOUND)
# if we have python:
#  add custom command that generates a html img comparison report for our tests
#  run with "make gen_report"
#  creates "_output/tout_img_report.html"
    add_custom_target(gen_report
                      ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/gen_img_compare_html_report.py
                      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                      COMMENT "Generating html summary of image compare tests")
endif()
