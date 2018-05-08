#######
# using gcc@4.9.2 compiler spec
#######

# c compiler used by spack
set("CMAKE_C_COMPILER" "/usr/apps/gnu/4.9.2/bin/gcc" CACHE PATH "")

# cpp compiler used by spack
set("CMAKE_CXX_COMPILER" "/usr/apps/gnu/4.9.2/bin/g++" CACHE PATH "")

# OPENMP Support
set("ENABLE_OPENMP" "ON" CACHE PATH "")

# CUDA support
set("ENABLE_CUDA" "ON" CACHE PATH "")

set("CUDA_BIN_DIR" "/opt/cudatoolkit-8.0/bin" CACHE PATH "")

# icet from uberenv
set("UMPIRE_DIR" "/usr/workspace/wsb/larsen30/surface/devil_ray/Umpire/install" CACHE PATH "")
set("RAJA_DIR" "/usr/workspace/wsb/larsen30/surface/devil_ray/RAJA/install" CACHE PATH "")


##################################
# end uberenv host-config
##################################
