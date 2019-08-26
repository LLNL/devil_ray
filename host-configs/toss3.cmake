#######
# using gcc@4.9.2 compiler spec
#######

# c compiler used by spack
set("CMAKE_C_COMPILER" "/usr/tce/bin/gcc" CACHE PATH "")

# cpp compiler used by spack
set("CMAKE_CXX_COMPILER" "/usr/tce/bin/g++" CACHE PATH "")

# OPENMP Support
set("ENABLE_OPENMP" "ON" CACHE PATH "")
set("BUILD_SHARED_LIBS" "OFF" CACHE PATH "")

# CUDA support
set("ENABLE_CUDA" "ON" CACHE PATH "")

set("CUDA_BIN_DIR" "/usr/tce/packages/cuda/cuda-9.1.85/" CACHE PATH "")
set("CUDA_ARCH" "sm_60" CACHE PATH "")


set("UMPIRE_DIR" "/usr/workspace/larsen30/pascal/devil_ray/2019/Umpire/install" CACHE PATH "")
set("RAJA_DIR" "/usr/workspace/larsen30/pascal/devil_ray/2019/RAJA/install" CACHE PATH "")

# MFEM support
set("ENABLE_MFEM" "ON" CACHE PATH "")
set("MFEM_DIR" "/usr/workspace/larsen30/pascal/devil_ray/2019/mfem/install" CACHE PATH "")
set("CONDUIT_DIR" "/usr/workspace/wsb/larsen30/pascal/devil_ray/2019/conduit/install" CACHE PATH "")
set("HDF5_DIR" "/usr/WS2/larsen30/pascal/devil_ray/2019/conduit/uberenv_libs/spack/opt/spack/linux-rhel7-x86_64/gcc-4.9.3/hdf5-1.8.21-3ukkkbqnx7sq4j6v3v75fhp5eans35ts" CACHE PATH "")


##################################
# end uberenv host-config
##################################
