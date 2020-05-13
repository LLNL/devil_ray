# Copyright 2013-2019 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)


from spack import *


class ApCompositor(CMakePackage):
    """A multi use-case image compositor"""

    homepage = 'https://github.com/Alpine-DAV/ap_compositor'
    git      = 'https://github.com/Alpine-DAV/ap_compositor.git'

    version('master', branch='master', submodules='True')

    variant('openmp', default=True, description='Build with openmp support')
    variant('mpi', default=True, description='Build with openmp support')
    variant('shared', default=True, description='Build Shared Library')

    depends_on('cmake@3.9:', type='build')

    root_cmakelists_dir = 'src'

    def cmake_args(self):
        spec = self.spec

        options = []

        if '+shared' in spec:
            options.append('-DBUILD_SHARED_LIBS=On')
        else:
            options.append('-DBUILD_SHARED_LIBS=Off')

        if '+openmp' in spec:
            options.append('-DENABLE_OPENMP=On')
        else:
            options.append('-DENABLE_OPENMP=Off')

        if '+mpi' in spec:
            options.append('-DENABLE_MPI=On')
        else:
            options.append('-DENABLE_MPI=Off')

        return options
