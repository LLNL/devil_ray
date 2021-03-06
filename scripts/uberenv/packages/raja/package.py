# Copyright 2013-2019 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

from spack import *


class Raja(CMakePackage,CudaPackage):
    """RAJA Parallel Framework."""

    homepage = "http://software.llnl.gov/RAJA/"
    git      = "https://github.com/LLNL/RAJA.git"

    version('develop', branch='develop', submodules='True')
    version('main',  branch='main',  submodules='True')
    version('0.12.1', tag='v0.12.1', submodules="True")
    version('0.12.0', tag='v0.12.0', submodules="True")
    version('0.11.0', tag='v0.11.0', submodules="True")
    version('0.10.1', tag='v0.10.1', submodules="True")
    version('0.10.0', tag='v0.10.0', submodules="True")
    version('0.9.0', tag='v0.9.0', submodules="True")
    version('0.8.0', tag='v0.8.0', submodules="True")
    version('0.7.0', tag='v0.7.0', submodules="True")
    version('0.6.0', tag='v0.6.0', submodules="True")
    version('0.5.3', tag='v0.5.3', submodules="True")
    version('0.5.2', tag='v0.5.2', submodules="True")
    version('0.5.1', tag='v0.5.1', submodules="True")
    version('0.5.0', tag='v0.5.0', submodules="True")
    version('0.4.1', tag='v0.4.1', submodules="True")
    version('0.4.0', tag='v0.4.0', submodules="True")

    variant('cuda', default=False, description='Build with CUDA backend')
    variant('openmp', default=True, description='Build OpenMP backend')
    variant('shared', default=True, description='Build Shared Library')
    variant('examples', default=False, description='Build examples.')
    variant('exercises', default=False, description='Build exercises.')

    depends_on('cuda', when='+cuda')

    depends_on('cmake@3.8:', type='build')
    depends_on('cmake@3.9:', when='+cuda', type='build')

    def cmake_args(self):
        spec = self.spec

        options = []

        if '+openmp' in spec:
            options.extend([
                '-DENABLE_OPENMP=On'])
        else:
            options.extend([
                '-DENABLE_OPENMP=OFF'])

        if '+cuda' in spec:
            options.extend([
                '-DENABLE_CUDA=On',
                '-DCUDA_TOOLKIT_ROOT_DIR=%s' % (spec['cuda'].prefix)])
            if 'cuda_arch' in spec.variants:
              cuda_value = spec.variants['cuda_arch'].value
              cuda_arch = cuda_value[0]
              options.append('-DCUDA_ARCH=sm_{0}'.format(cuda_arch))

        if '+shared' in spec:
            options.append('-DBUILD_SHARED_LIBS=On')
        else:
            options.append('-DBUILD_SHARED_LIBS=Off')

        options.append('-DENABLE_EXAMPLES={0}'.format(
            'ON' if '+examples' in spec else 'OFF'))

        options.append('-DENABLE_EXERCISES={0}'.format(
            'ON' if '+exercises' in spec else 'OFF'))

        # Work around spack adding -march=ppc64le to SPACK_TARGET_ARGS which
        # is used by the spack compiler wrapper.  This can go away when BLT
        # removes -Werror from GTest flags
        if self.spec.satisfies('%clang target=ppc64le:') or not self.run_tests:
            options.append('-DENABLE_TESTS=OFF')
        else:
            options.append('-DENABLE_TESTS=ON')

        return options
