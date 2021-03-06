set(dray_headers dray.hpp
                 array.hpp
                 aabb.hpp
                 additional_color_tables.hpp
                 array_internals.hpp
                 array_internals_base.hpp
                 array_registry.hpp
                 array_utils.hpp
                 color_map.hpp
                 color_table.hpp
                 dray_node_to_dataset.hpp
                 exports.hpp
                 error.hpp
                 error_check.hpp
                 halton.hpp
                 linear_bvh_builder.hpp
                 location.hpp
                 math.hpp
                 integer_utils.hpp
                 morton_codes.hpp
                 matrix.hpp
                 newton_solver.hpp
                 subdivision_search.hpp
                 plane_detector.hpp
                 policies.hpp
                 power_basis.hpp
                 random.hpp
                 range.hpp
                 ray.hpp
                 ray_hit.hpp
                 ref_point.hpp
                 simple_tensor.hpp
                 template_tag.hpp
                 types.hpp
                 transform_3d.hpp
                 triangle_mesh.hpp
                 triangle_intersection.hpp
                 vec.hpp
                 warning.hpp
                 io/obj_reader.hpp
                 io/mfem_reader.hpp
                 io/blueprint_reader.hpp
                 io/blueprint_low_order.hpp
                 import_order_policy.hpp

                 rendering/annotator.hpp
                 rendering/camera.hpp
                 rendering/colors.hpp
                 rendering/contour.hpp
                 rendering/color_bar_annotator.hpp
                 rendering/device_framebuffer.hpp
                 rendering/font.hpp
                 rendering/font_factory.hpp
                 rendering/fragment.hpp
                 rendering/framebuffer.hpp
                 rendering/low_order_intersectors.hpp
                 rendering/point_light.hpp
                 rendering/traceable.hpp
                 rendering/renderer.hpp
                 rendering/scalar_buffer.hpp
                 rendering/scalar_renderer.hpp
                 rendering/slice_plane.hpp
                 rendering/surface.hpp
                 rendering/text_annotator.hpp
                 rendering/volume.hpp
                 rendering/volume_partial.hpp
                 rendering/volume_shader.hpp

                 queries/lineout.hpp
                 queries/point_location.hpp

                 utils/color_buffer_utils.hpp
                 utils/data_logger.hpp
                 utils/png_encoder.hpp
                 utils/png_decoder.hpp
                 utils/png_compare.hpp
                 utils/ray_utils.hpp
                 utils/string_utils.hpp
                 utils/mpi_utils.hpp
                 utils/timer.hpp
                 utils/yaml_writer.hpp

                 ambient_occlusion.hpp
                 intersection_context.hpp
                 binomial.hpp
                 constants.hpp
                 utils/stats.hpp
                 utils/appstats.hpp
                 utils/dataset_builder.hpp

                 dispatcher.hpp

                 data_model/collection.hpp
                 data_model/data_set.hpp
                 data_model/bernstein_basis.hpp
                 data_model/bezier_simplex.hpp
                 data_model/subref.hpp
                 data_model/elem_attr.hpp
                 data_model/ref_space.hpp
                 data_model/element.hpp
                 data_model/detached_element.hpp
                 data_model/elem_utils.hpp
                 data_model/elem_ops.hpp
                 data_model/iso_ops.hpp
                 data_model/pos_tensor_element.tcc
                 data_model/pos_simplex_element.tcc
                 data_model/subpatch.hpp
                 data_model/mesh_utils.hpp
                 data_model/field.hpp
                 data_model/unstructured_field.hpp
                 data_model/grid_function.hpp
                 data_model/unstructured_mesh.hpp
                 data_model/mesh.hpp

                 filters/mesh_boundary.hpp
                 filters/reflect.hpp
                 filters/redistribute.hpp
                 filters/subset.hpp
                 filters/volume_balance.hpp
                 filters/isosurfacing.hpp
                 filters/to_bernstein.hpp
                 filters/vector_component.hpp

                 #filters/attractor_map.hpp
                 filters/surface_triangle.hpp

                 synthetic/affine_radial.hpp
                 synthetic/spiral_sample.hpp
                 synthetic/tet_sphere_sample.hpp
)

set(dray_sources dray.cpp
                 array.cpp
                 array_internals.cpp
                 array_internals_base.cpp
                 array_registry.cpp
                 color_map.cpp
                 color_table.cpp
                 dray_node_to_dataset.cpp
                 linear_bvh_builder.cpp
                 location.cpp
                 matrix.cpp
                 subdivision_search.cpp
                 ray.cpp
                 ray_hit.cpp
                 triangle_mesh.cpp
                 warning.cpp
                 vec.cpp
                 utils/color_buffer_utils.cpp
                 utils/data_logger.cpp
                 utils/png_encoder.cpp
                 utils/png_decoder.cpp
                 utils/png_compare.cpp
                 utils/ray_utils.cpp
                 utils/string_utils.cpp
                 utils/mpi_utils.cpp

                 queries/lineout.cpp
                 queries/point_location.cpp

                 ambient_occlusion.cpp
                 intersection_context.cpp
                 binomial.cpp

                 io/mfem_reader.cpp
                 io/blueprint_reader.cpp
                 io/blueprint_low_order.cpp

                 rendering/annotator.cpp
                 rendering/camera.cpp
                 rendering/color_bar_annotator.cpp
                 rendering/contour.cpp
                 rendering/font.cpp
                 rendering/font_factory.cpp
                 rendering/fragment.cpp
                 rendering/framebuffer.cpp
                 rendering/traceable.cpp
                 rendering/point_light.cpp
                 rendering/renderer.cpp
                 rendering/scalar_buffer.cpp
                 rendering/scalar_renderer.cpp
                 rendering/slice_plane.cpp
                 rendering/surface.cpp
                 rendering/text_annotator.cpp
                 rendering/volume.cpp
                 rendering/volume_partial.cpp

                 utils/appstats.cpp
                 utils/dataset_builder.cpp

                 dispatcher.cpp

                 data_model/collection.cpp
                 data_model/data_set.cpp
                 data_model/element.cpp
                 data_model/pos_tensor_element.cpp
                 data_model/pos_simplex_element.cpp
                 data_model/subref.cpp
                 data_model/iso_ops.cpp
                 data_model/grid_function.cpp
                 data_model/unstructured_mesh.cpp
                 data_model/mesh_utils.cpp
                 data_model/unstructured_field.cpp

                 filters/mesh_boundary.cpp
                 filters/reflect.cpp
                 filters/redistribute.cpp
                 filters/subset.cpp
                 filters/volume_balance.cpp
                 filters/isosurfacing.cpp
                 filters/to_bernstein.cpp
                 filters/vector_component.cpp

                 #filters/attractor_map.cpp
                 filters/surface_triangle.cpp

                 synthetic/affine_radial.cpp
                 synthetic/spiral_sample.cpp
                 synthetic/tet_sphere_sample.cpp
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

  if(NOT CUDA_ARCH)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_60,code=sm_60")
  else()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch ${CUDA_ARCH}")
  endif()
  # make sure we propagate host compilers flags
  #if(ENABLE_OPENMP)
  #  list(APPEND dray_cuda_flags "-Xcompiler -fopenmp")
  #endif()

  # enable flag to run on the CPU without rebuilding everything
  if(DEBUG_CPU_ONLY)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DDEBUG_CPU_ONLY")
  endif()

  # Building the BVH requires us to propagate bounding boxes in a tree
  # from the bottom up. This means one thread can update aabbs in a node
  # before terminating and letting another thread take over processing
  # of the sub-tree. That said, there are no promises about which thread,
  # and apparently there is no cache coherence between L1 caches so the
  # other thread can might not get the updated data, destroying entire
  # subtrees of the bvh. This doesn't happen vary often, but does happen.
  # This compile flags disables L1 cache for this source file, preventing
  # the issue. This line is the culmination of 10 hours of debugging,
  # during which I questioned my own reality (ML). Slight optimization
  # would be to move the one function to another file.
  set_source_files_properties(linear_bvh_builder.cpp PROPERTIES COMPILE_FLAGS "-Xptxas -dlcm=cg")
endif()

#################################e#######
# Serial Devil Ray (i.e., no mpi)
########################################
if(ENABLE_SERIAL)
  blt_add_library(
    NAME dray
    SOURCES ${dray_sources}
    HEADERS ${dray_headers}
    DEPENDS_ON ${dray_thirdparty_libs} apcomp
  )

########################################
# Compling flags
########################################
  if(ENABLE_LOGGING)
    blt_add_target_compile_flags(TO dray FLAGS " -D DRAY_ENABLE_LOGGING=1")
    message(STATUS "Logging enabled")
  endif()

  if(DRAY_USE_DOUBLE_PRECISION)
    blt_add_target_compile_flags(TO dray FLAGS " -D DRAY_DOUBLE_PRECISION=1")
    message(STATUS "Using double precision")
  endif()

  if(ENABLE_OPENMP)
    blt_add_target_compile_flags(TO dray FLAGS " -D DRAY_OPENMP_ENABLED=1")
  endif()

  if(ENABLE_STATS)
    blt_add_target_compile_flags(TO dray FLAGS " -D DRAY_STATS=1")
    message(STATUS "Stats enabled")
  endif()

  set_target_properties(dray PROPERTIES
                        CXX_STANDARD 11
                        CXX_STANDARD_REQUIRED YES
                        CXX_EXTENTIONS NO)

  get_target_property(MAIN_CFLAGS dray COMPILE_OPTIONS)
  message("-- Target compiler flags are: ${MAIN_CFLAGS}")

  target_include_directories(dray PRIVATE ../${CMAKE_CURRENT_SOURCE_DIR} )

  install(TARGETS dray
          EXPORT dray
          LIBRARY DESTINATION lib
          ARCHIVE DESTINATION lib
          RUNTIME DESTINATION lib
  )
endif()

if(MPI_FOUND)
  ########################################
  # MPI Devil Ray
  ########################################
  set(dray_mpi_deps mpi ${dray_thirdparty_libs} conduit_relay_mpi)

  blt_add_library(
    NAME dray_mpi
    SOURCES ${dray_sources}
    HEADERS ${dray_headers}
    DEPENDS_ON ${dray_mpi_deps} apcomp_mpi
    )

  ########################################
  # Compling flags
  ########################################
  if(ENABLE_LOGGING)
    blt_add_target_compile_flags(TO dray_mpi FLAGS " -D DRAY_ENABLE_LOGGING=1")
    message(STATUS "Logging enabled")
  endif()

  if(DRAY_USE_DOUBLE_PRECISION)
    blt_add_target_compile_flags(TO dray_mpi FLAGS " -D DRAY_DOUBLE_PRECISION=1")
    message(STATUS "Using double precision")
  endif()

  if(ENABLE_OPENMP)
    blt_add_target_compile_flags(TO dray_mpi FLAGS " -D DRAY_OPENMP_ENABLED=1")
  endif()

  if(ENABLE_STATS)
    blt_add_target_compile_flags(TO dray_mpi FLAGS " -D DRAY_STATS=1")
    message(STATUS "Stats enabled")
  endif()

  target_compile_definitions(dray_mpi PRIVATE DRAY_MPI_ENABLED)

  set_target_properties(dray_mpi PROPERTIES
                        CXX_STANDARD 11
                        CXX_STANDARD_REQUIRED YES
                        CXX_EXTENTIONS NO)

  target_include_directories(dray_mpi PRIVATE ../${CMAKE_CURRENT_SOURCE_DIR} )

  install(TARGETS dray_mpi
          EXPORT dray
          LIBRARY DESTINATION lib
          ARCHIVE DESTINATION lib
          RUNTIME DESTINATION lib)
endif()

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
