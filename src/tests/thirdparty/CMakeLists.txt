
message(STATUS "Adding Umpire smoke test")
add_cpp_test(TEST t_umpire_smoke DEPENDS_ON umpire)

if(ENABLE_MFEM)
  add_cpp_test(TEST t_mfem_smoke DEPENDS_ON mfem)
endif()
