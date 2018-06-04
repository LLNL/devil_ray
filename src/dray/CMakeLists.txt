set(dray_headers dray.hpp
                 array.hpp
                 aabb.hpp
                 array_internals.hpp
                 array_internals_base.hpp
                 array_registry.hpp
                 array_utils.hpp
                 camera.hpp
                 linear_bvh_builder.hpp
                 math.hpp
                 morton_codes.hpp
                 point_location.hpp
                 range.hpp
                 ray.hpp
                 test.hpp
                 triangle_mesh.hpp
                 triangle_intersection.hpp
                 vec.hpp
                 io/obj_reader.hpp
                 utils/data_logger.hpp
                 utils/png_encoder.hpp
                 utils/ray_utils.hpp
                 utils/timer.hpp
                 utils/yaml_writer.hpp
)

set(dray_sources dray.cpp
                 array.cpp
                 array_internals.cpp
                 array_internals_base.cpp
                 array_registry.cpp
                 camera.cpp
                 linear_bvh_builder.cpp
                 point_location.cpp
                 ray.cpp
                 triangle_mesh.cpp
                 vec.cpp
                 test.cpp
                 utils/data_logger.cpp
                 utils/png_encoder.cpp
)


set(dray_mfem_sources mfem_mesh.cpp
)

set(dray_mfem_headers mfem_mesh.hpp
)
################################################
# MFEM support
################################################
if(ENABLE_MFEM)
  list(APPEND dray_thirdparty_libs mfem)
  list(APPEND dray_sources ${dray_mfem_sources})
  list(APPEND dray_headers ${dray_mfem_headers})
endif()

################################################
# third party required deps 
################################################
set(dray_thirdparty_libs RAJA umpire lodepng)

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

  set(CUDA_NVCC_FLAGS_PREV ${CUDA_NVCC_FLAGS})
  set(dray_cuda_flags ${RAJA_NVCC_FLAGS}) 
  # make sure we propagate host compilers flags
  if(ENABLE_OPENMP)
    list(APPEND dray_cuda_flags "-Xcompiler -fopenmp")    
  endif()
  
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} ${dray_cuda_flags}")

endif()

blt_add_library(
  NAME dray
  SOURCES ${dray_sources}
  HEADERS ${dray_headers}
  DEPENDS_ON ${dray_thirdparty_libs}
)

message(STATUS "adding dray lib ${dray_sources} ${dray_headers}")

if(ENABLE_CUDA)
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS_PREV})
endif()

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
