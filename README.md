Building Devil Ray
------------------
Devil Ray uses a Spack-based build system.
We include several Spack configuration directories that include
both OpenMP and CUDA configs.
We require CMake 3.9 or greater when building with OpenMP
and CMake 3.14 or greater when building with CUDA.
Example of building on LLNL's Pascal cluster:

```
git clone --recursive https://github.com/LLNL/devil_ray.git
cd devil_ray
python ./scripts/uberenv/uberenv.py --spack-config-dir=./scripts/uberenv/spack_configs/llnl/pascal_cuda/
mkdir build
cd build
ml cmake
ml cuda
cmake -C ../uberenv_libs/pascal1-toss_3_x86_64_ib-gcc@4.9.3-ascent.cmake ../src/
make
```

License
----------------

Devil Ray is distributed under the terms of BSD-3-Clause

All new contributions must be made under the BSD-3-Clause
license

SPDX-License-Identifier: (BSD-3-Clause)

LLNL-CODE-797171
