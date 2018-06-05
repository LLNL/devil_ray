
set(DATA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/data/")
configure_file(test_config.h.in "${CMAKE_CURRENT_SOURCE_DIR}/test_config.h" @ONLY)

set(BASIC_TESTS t_dray_smoke
                t_dray_array
                t_dray_obj_reader
                t_dray_registry
                t_dray_tri_benchmark
                t_dray_test
                t_dray_vec
		t_dray_ambient_occlusion
)

message(STATUS "Adding basic dray unit tests")
foreach(TEST ${BASIC_TESTS})
  message(STATUS "adding test ${TEST}")
  add_cpp_test(TEST ${TEST} DEPENDS_ON dray mfem)
endforeach()

if(ENABLE_MFEM)
  set(MFEM_TESTS t_dray_mfem
  )
  message(STATUS "Adding mfem dray unit tests")
  foreach(TEST ${MFEM_TESTS})
    message(STATUS "adding test ${TEST}")
    add_cpp_test(TEST ${TEST} DEPENDS_ON dray mfem)
  endforeach()
endif()
