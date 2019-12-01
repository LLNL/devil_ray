set(dray_headers dray.hpp
                 array.hpp
                 aabb.hpp
                 array_internals.hpp
                 array_internals_base.hpp
                 array_registry.hpp
                 array_utils.hpp
                 camera.hpp
                 color_map.hpp
                 color_table.hpp
                 data_set.hpp
                 new_data_set.hpp
                 derived_field.hpp
                 derived_topology.hpp
                 device_framebuffer.hpp
                 exports.hpp
                 error.hpp
                 framebuffer.hpp
                 halton.hpp
                 linear_bvh_builder.hpp
                 location.hpp
                 math.hpp
                 integer_utils.hpp
                 morton_codes.hpp
                 matrix.hpp
                 newton_solver.hpp
                 subdivision_search.hpp
                 point_location.hpp
                 policies.hpp
                 power_basis.hpp
                 range.hpp
                 ray.hpp
                 ray_hit.hpp
                 ref_point.hpp
                 simple_tensor.hpp
                 shaders.hpp
                 topology_base.hpp
                 test.hpp
                 types.hpp
                 transform_3d.hpp
                 triangle_mesh.hpp
                 triangle_intersection.hpp
                 vec.hpp
                 io/obj_reader.hpp
                 io/mfem_reader.hpp
                 io/blueprint_reader.hpp
                 utils/color_buffer_utils.hpp
                 utils/data_logger.hpp
                 utils/png_encoder.hpp
                 utils/png_decoder.hpp
                 utils/png_compare.hpp
                 utils/ray_utils.hpp
                 utils/timer.hpp
                 utils/yaml_writer.hpp

                 ambient_occlusion.hpp
                 intersection_context.hpp
                 binomial.hpp
                 constants.hpp
                 utils/stats.hpp
                 utils/global_share.hpp
                 utils/appstats.hpp

                 Element/bernstein_basis.hpp
                 Element/bezier_simplex.hpp
                 Element/element.hpp
                 Element/elem_utils.hpp
                 Element/pos_tensor_element.tcc
                 Element/pos_simplex_element.tcc
                 Element/subpatch.hpp
                 GridFunction/mesh.hpp
                 GridFunction/mesh_utils.hpp
                 GridFunction/field.hpp
                 GridFunction/grid_function.hpp

                 filters/isosurface.hpp
                 filters/slice.hpp
                 filters/mesh_lines.hpp
                 filters/mesh_boundary.hpp
                 filters/volume_integrator.hpp

                 filters/attractor_map.hpp
                 filters/surface_triangle.hpp

                 filters/internal/get_fragments.hpp
)

set(dray_sources dray.cpp
                 array.cpp
                 array_internals.cpp
                 array_internals_base.cpp
                 array_registry.cpp
                 camera.cpp
                 color_map.cpp
                 color_table.cpp
                 data_set.cpp
                 new_data_set.cpp
                 derived_field.cpp
                 derived_topology.cpp
                 framebuffer.cpp
                 linear_bvh_builder.cpp
                 matrix.cpp
                 subdivision_search.cpp
                 point_location.cpp
                 ray.cpp
                 shaders.cpp
                 triangle_mesh.cpp
                 vec.cpp
                 test.cpp
                 utils/color_buffer_utils.cpp
                 utils/data_logger.cpp
                 utils/png_encoder.cpp
                 utils/png_decoder.cpp
                 utils/png_compare.cpp
                 utils/ray_utils.cpp
                 ambient_occlusion.cpp
                 intersection_context.cpp
                 binomial.cpp

                 io/mfem_reader.cpp
                 io/blueprint_reader.cpp

                 utils/appstats.cpp

                 Element/element.cpp
                 Element/pos_tensor_element.cpp
                 Element/pos_simplex_element.cpp
                 GridFunction/grid_function.cpp
                 GridFunction/mesh.cpp
                 GridFunction/mesh_utils.cpp
                 GridFunction/field.cpp

                 filters/isosurface.cpp
                 filters/mesh_lines.cpp
                 filters/mesh_boundary.cpp
                 filters/slice.cpp

                 filters/attractor_map.cpp
                 filters/surface_triangle.cpp

                 filters/volume_integrator.cpp
                 filters/internal/get_fragments.cpp
)


set(dray_mfem_sources utils/mfem_utils.cpp
                      mfem2dray.cpp
)

set(dray_mfem_headers utils/mfem_utils.hpp
                      mfem2dray.hpp
)
convert_to_native_escaped_file_path(${CMAKE_INSTALL_PREFIX}
                                    DRAY_INSTALL_PREFIX)

set(DRAY_INSTALL_PREFIX ${DRAY_INSTALL_PREFIX} CACHE STRING "" FORCE )
################################################
# third party required deps
################################################
set(dray_thirdparty_libs RAJA umpire dray_lodepng)

################################################
# third party required
################################################
list(APPEND dray_thirdparty_libs conduit conduit_relay)


################################################
# MFEM support
################################################
list(APPEND dray_thirdparty_libs mfem)
list(APPEND dray_sources ${dray_mfem_sources})
list(APPEND dray_headers ${dray_mfem_headers})

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

if(ENABLE_LOGGING)
  blt_add_target_compile_flags(TO dray FLAGS " -D DRAY_ENABLE_LOGGING=1")
  message(STATUS "Logging enabled")
endif()

if(DRAY_USE_DOUBLE_PRECISION)
  blt_add_target_compile_flags(TO dray FLAGS " -D DRAY_DOUBLE_PRECISION=1")
  message(STATUS "Using double precision")
endif()

if(ENABLE_OPENMP)
  blt_add_target_compile_flags(TO dray FLAGS " -D USE_OPENMP=1")
endif()

if(ENABLE_STATS)
  blt_add_target_compile_flags(TO dray FLAGS " -D DRAY_STATS=1")
  message(STATUS "Stats enabled")
endif()

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

#export(TARGETS dray FILE drayTargets.cmake)

########################################
# Preserve install directory structure
########################################
macro(install_headers_with_directory header_list)

  foreach(header ${${header_list}})
    string(REGEX MATCH "(.*)[/\\]" DIR ${header})
    install(FILES ${header} DESTINATION include/dray/${DIR})
  endforeach(header)

endmacro(install_headers_with_directory)
install_headers_with_directory(dray_headers)
