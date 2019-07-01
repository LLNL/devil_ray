set(dray_headers dray.hpp
                 array.hpp
                 aabb.hpp
                 array_internals.hpp
                 array_internals_base.hpp
                 array_registry.hpp
                 array_utils.hpp
                 bernstein_basis.hpp
                 camera.hpp
                 color_table.hpp
                 data_set.hpp
                 el_trans.hpp
                 error.hpp
                 linear_bvh_builder.hpp
                 math.hpp
                 morton_codes.hpp
                 matrix.hpp
                 newton_solver.hpp
                 subdivision_search.hpp
                 point_location.hpp
                 power_basis.hpp
                 range.hpp
                 ray.hpp
                 ref_point.hpp
                 simple_tensor.hpp
                 shaders.hpp
                 test.hpp
                 triangle_mesh.hpp
                 triangle_intersection.hpp
                 vec.hpp
                 io/obj_reader.hpp
                 io/mfem_reader.hpp
                 utils/data_logger.hpp
                 utils/png_encoder.hpp
                 utils/png_decoder.hpp
                 utils/png_compare.hpp
                 utils/ray_utils.hpp
                 utils/timer.hpp
                 utils/yaml_writer.hpp

                 ambient_occlusion.hpp
                 intersection_context.hpp
                 shading_context.hpp
                 binomial.hpp
                 constants.hpp
                 utils/stats.hpp
                 utils/global_share.hpp
                 utils/appstats.hpp

                 Element/element.hpp
                 Element/subpatch.hpp
                 GridFunction/mesh.hpp
                 GridFunction/field.hpp
                 GridFunction/grid_function_data.hpp

                 filters/isosurface.hpp
                 filters/slice.hpp
                 filters/mesh_lines.hpp
                 filters/volume_integrator.hpp
)

set(dray_sources dray.cpp
                 array.cpp
                 array_internals.cpp
                 array_internals_base.cpp
                 array_registry.cpp
                 camera.cpp
                 color_table.cpp
                 data_set.cpp
                 el_trans.cpp
                 linear_bvh_builder.cpp
                 matrix.cpp
                 point_location.cpp
                 ray.cpp
                 shaders.cpp
                 triangle_mesh.cpp
                 vec.cpp
                 test.cpp
                 utils/data_logger.cpp
                 utils/png_encoder.cpp
                 utils/png_decoder.cpp
                 utils/png_compare.cpp
                 ambient_occlusion.cpp
                 intersection_context.cpp
                 shading_context.cpp
                 binomial.cpp

                 io/mfem_reader.cpp

                 utils/appstats.cpp

                 Element/element.cpp
                 GridFunction/mesh.cpp
                 GridFunction/field.cpp

                 filters/isosurface.cpp
                 filters/mesh_lines.cpp
                 filters/slice.cpp
                 filters/volume_integrator.cpp
)


set(dray_mfem_sources mfem_mesh.cpp
                      mfem_data_set.cpp
                      mfem_volume_integrator.cpp
                      mfem_grid_function.cpp
                      utils/mfem_utils.cpp
                      mfem2dray.cpp
)

set(dray_mfem_headers mfem_mesh.hpp
                      mfem_data_set.cpp
                      mfem_volume_integrator.hpp
                      mfem_grid_function.hpp
                      utils/mfem_utils.hpp
                      mfem2dray.hpp
)

################################################
# third party required deps
################################################
set(dray_thirdparty_libs RAJA umpire lodepng)

################################################
# third party required
################################################
list(APPEND dray_thirdparty_libs conduit)


################################################
# MFEM support
################################################
if(ENABLE_MFEM)
  list(APPEND dray_thirdparty_libs mfem)
  list(APPEND dray_sources ${dray_mfem_sources})
  list(APPEND dray_headers ${dray_mfem_headers})
endif()

################################################
# openmp support
################################################
if(ENABLE_OPENMP)
  list(APPEND dray_thirdparty_libs openmp)
endif()

################################################
# cuda support
################################################
if(ENABLE_CUDA)
  list(APPEND dray_thirdparty_libs cuda)
endif()

if(ENABLE_CUDA)

  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda --expt-relaxed-constexpr")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_60,code=sm_60")
  # make sure we propagate host compilers flags
  if(ENABLE_OPENMP)
    list(APPEND dray_cuda_flags "-Xcompiler -fopenmp")
  endif()

  # enable flag to run on the CPU without rebuilding everything
  if(DEBUG_CPU_ONLY)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DDEBUG_CPU_ONLY")
  endif()

endif()

blt_add_library(
  NAME dray
  SOURCES ${dray_sources}
  HEADERS ${dray_headers}
  DEPENDS_ON ${dray_thirdparty_libs}
)

message(STATUS "adding dray lib ${dray_sources} ${dray_headers}")


set_target_properties(dray PROPERTIES
                      CXX_STANDARD 11
                      CXX_STANDARD_REQUIRED YES
                      CXX_EXTENTIONS NO)

target_include_directories(dray PRIVATE ../${CMAKE_CURRENT_SOURCE_DIR} )

install(TARGETS dray
        EXPORT dray
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION lib
)

install(FILES ${dray_headers} DESTINATION include/dray)
