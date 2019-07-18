# OPENMP Support
set("BUILD_SHARED_LIBS" "OFF" CACHE PATH "")
set("ENABLE_OPENMP" "ON" CACHE PATH "")

# CUDA support
set("ENABLE_CUDA" "OFF" CACHE PATH "")

set(CMAKE_C_COMPILER "/opt/local/bin/gcc-mp-8" CACHE PATH "")
set(CMAKE_CXX_COMPILER "/opt/local/bin/g++-mp-8" CACHE PATH "")

set(HDF5_DIR "/Users/larsen30/research/devil_ray/hdf5-1.8.21/install" CACHE PATH "")
set(CONDUIT_DIR "/Users/larsen30/research/devil_ray/conduit/install" CACHE PATH "")
set(UMPIRE_DIR "/Users/larsen30/research/devil_ray/Umpire/install" CACHE PATH "")
set(RAJA_DIR "/Users/larsen30/research/devil_ray/RAJA/install" CACHE PATH "")
set(MFEM_DIR "/Users/larsen30/research/devil_ray/mfem/install_dir" CACHE PATH "")

