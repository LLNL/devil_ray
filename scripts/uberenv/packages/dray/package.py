# Copyright 2013-2019 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

from spack import *

import sys
import os
import socket
import glob
import shutil

import llnl.util.tty as tty
from os import environ as env

def cmake_cache_entry(name, value, vtype=None):
    """
    Helper that creates CMake cache entry strings used in
    'host-config' files.
    """
    if vtype is None:
        if value == "ON" or value == "OFF":
            vtype = "BOOL"
        else:
            vtype = "PATH"
    return 'set({0} "{1}" CACHE {2} "")\n\n'.format(name, value, vtype)

class Dray(Package):
    """High-Order Mesh Ray Tracer."""

    homepage = "https://lc.llnl.gov/bitbucket/projects/VIS/repos/devil_ray/browse"
    git      = "ssh://git@cz-bitbucket.llnl.gov:7999/vis/devil_ray.git"

    version('master',  branch='master',  submodules='True')

    variant('cuda', default=False, description='Build with CUDA backend')
    variant('openmp', default=True, description='Build OpenMP backend')
    variant("shared", default=True, description="Build as shared libs")
    variant("test", default=True, description='Build unit tests')
    variant("logging", default=False, description='Enable logging')
    variant("stats", default=False, description='Enable stats')

    depends_on('cuda', when='+cuda')

    depends_on('cmake@3.8:', type='build')
    depends_on('cmake@3.9:', when='+cuda', type='build')

    depends_on("conduit~shared~mpi~python", when="~shared")
    depends_on("conduit+shared~mpi~python", when="+shared")

    depends_on("raja@0.9.0+cuda~openmp", when="+cuda~openmp")
    depends_on("raja@0.9.0+cuda+openmp", when="+cuda+openmp")
    depends_on("raja@0.9.0+cuda~openmp~shared", when="+dray+cuda~openmp~shared")
    depends_on("raja@0.9.0+cuda+openmp~shared", when="+dray+cuda+openmp~shared")

    depends_on("raja@0.9.0~cuda~openmp", when="~cuda~openmp")
    depends_on("raja@0.9.0~cuda+openmp", when="~cuda+openmp")
    depends_on("raja@0.9.0~cuda~openmp~shared", when="~cuda~openmp~shared")
    depends_on("raja@0.9.0~cuda+openmp~shared", when="~cuda+openmp~shared")

    depends_on("umpire@1.0.0+cuda", when="+cuda")
    depends_on("umpire@1.0.0+cuda~shared", when="+cuda~shared")
    depends_on("umpire@1.0.0~cuda", when="~cuda")
    depends_on("umpire@1.0.0~cuda~shared", when="~cuda~shared")

    depends_on("mfem~mpi+shared+conduit+threadsafe", when="+shared")
    depends_on("mfem~mpi~shared+conduit+threadsafe", when="~shared")

    def setup_environment(self, spack_env, run_env):
        spack_env.set('CTEST_OUTPUT_ON_FAILURE', '1')

    def install(self, spec, prefix):
        """
        Build and install Devil Ray.
        """
        with working_dir('spack-build', create=True):
            py_site_pkgs_dir = None

            host_cfg_fname = self.create_host_config(spec,
                                                     prefix,
                                                     py_site_pkgs_dir)
            cmake_args = []
            # if we have a static build, we need to avoid any of
            # spack's default cmake settings related to rpaths
            # (see: https://github.com/LLNL/spack/issues/2658)
            if "+shared" in spec:
                cmake_args.extend(std_cmake_args)
            else:
                for arg in std_cmake_args:
                    if arg.count("RPATH") == 0:
                        cmake_args.append(arg)
            cmake_args.extend(["-C", host_cfg_fname, "../src"])
            print("Configuring Devil Ray...")
            cmake(*cmake_args)
            print("Building Devil Ray...")
            make()
            # run unit tests if requested
            if "+test" in spec and self.run_tests:
                print("Running Devil Ray Unit Tests...")
                make("test")
            print("Installing Devil Ray...")
            make("install")
            # install copy of host config for provenance
            install(host_cfg_fname, prefix)

    def create_host_config(self, spec, prefix, py_site_pkgs_dir=None):
        """
        This method creates a 'host-config' file that specifies
        all of the options used to configure and build ascent.

        For more details about 'host-config' files see:
            http://ascent.readthedocs.io/en/latest/BuildingAscent.html

        Note:
          The `py_site_pkgs_dir` arg exists to allow a package that
          subclasses this package provide a specific site packages
          dir when calling this function. `py_site_pkgs_dir` should
          be an absolute path or `None`.

          This is necessary because the spack `site_packages_dir`
          var will not exist in the base class. For more details
          on this issue see: https://github.com/spack/spack/issues/6261
        """

        #######################
        # Compiler Info
        #######################
        c_compiler = env["SPACK_CC"]
        cpp_compiler = env["SPACK_CXX"]
        f_compiler = None

        if self.compiler.fc:
            # even if this is set, it may not exist so do one more sanity check
            f_compiler = env["SPACK_FC"]

        #######################################################################
        # By directly fetching the names of the actual compilers we appear
        # to doing something evil here, but this is necessary to create a
        # 'host config' file that works outside of the spack install env.
        #######################################################################

        sys_type = spec.architecture
        # if on llnl systems, we can use the SYS_TYPE
        if "SYS_TYPE" in env:
            sys_type = env["SYS_TYPE"]

        ##############################################
        # Find and record what CMake is used
        ##############################################

        if "+cmake" in spec:
            cmake_exe = spec['cmake'].command.path
        else:
            cmake_exe = which("cmake")
            if cmake_exe is None:
                msg = 'failed to find CMake (and cmake variant is off)'
                raise RuntimeError(msg)
            cmake_exe = cmake_exe.path

        host_cfg_fname = "%s-%s-%s-ascent.cmake" % (socket.gethostname(),
                                                    sys_type,
                                                    spec.compiler)

        cfg = open(host_cfg_fname, "w")
        cfg.write("##################################\n")
        cfg.write("# spack generated host-config\n")
        cfg.write("##################################\n")
        cfg.write("# {0}-{1}\n".format(sys_type, spec.compiler))
        cfg.write("##################################\n\n")

        # Include path to cmake for reference
        cfg.write("# cmake from spack \n")
        cfg.write("# cmake executable path: %s\n\n" % cmake_exe)

        #######################
        # Compiler Settings
        #######################
        cfg.write("#######\n")
        cfg.write("# using %s compiler spec\n" % spec.compiler)
        cfg.write("#######\n\n")
        cfg.write("# c compiler used by spack\n")
        cfg.write(cmake_cache_entry("CMAKE_C_COMPILER", c_compiler))
        cfg.write("# cpp compiler used by spack\n")
        cfg.write(cmake_cache_entry("CMAKE_CXX_COMPILER", cpp_compiler))

        #######################
        # Backends
        #######################

        cfg.write("# CUDA Support\n")

        if "+cuda" in spec:
            cfg.write(cmake_cache_entry("ENABLE_CUDA", "ON"))
        else:
            cfg.write(cmake_cache_entry("ENABLE_CUDA", "OFF"))

        if "+openmp" in spec:
            cfg.write(cmake_cache_entry("ENABLE_OPENMP", "ON"))
        else:
            cfg.write(cmake_cache_entry("ENABLE_OPENMP", "OFF"))

        # shared vs static libs
        if "+shared" in spec:
            cfg.write(cmake_cache_entry("BUILD_SHARED_LIBS", "ON"))
        else:
            cfg.write(cmake_cache_entry("BUILD_SHARED_LIBS", "OFF"))

        #######################
        # Unit Tests
        #######################
        if "+test" in spec:
            cfg.write(cmake_cache_entry("DRAY_ENABLE_TESTS", "ON"))
        else:
            cfg.write(cmake_cache_entry("DRAY_ENABLE_TESTS", "OFF"))

        #######################
        # Logging
        #######################
        if "+logging" in spec:
            cfg.write(cmake_cache_entry("ENABLE_LOGGING", "ON"))
        else:
            cfg.write(cmake_cache_entry("ENABLE_LOGGING", "OFF"))

        #######################
        # Logging
        #######################
        if "+stats" in spec:
            cfg.write(cmake_cache_entry("ENABLE_STATS", "ON"))
        else:
            cfg.write(cmake_cache_entry("ENABLE_STATS", "OFF"))

        #######################################################################
        # Core Dependencies
        #######################################################################

        cfg.write("# conduit from spack \n")
        cfg.write(cmake_cache_entry("CONDUIT_DIR", spec['conduit'].prefix))

        cfg.write("# mfem from spack \n")
        cfg.write(cmake_cache_entry("MFEM_DIR", spec['mfem'].prefix))

        cfg.write("# raja from spack \n")
        cfg.write(cmake_cache_entry("RAJA_DIR", spec['raja'].prefix))

        cfg.write("# umpire from spack \n")
        cfg.write(cmake_cache_entry("UMPIRE_DIR", spec['umpire'].prefix))

        cfg.write("##################################\n")
        cfg.write("# end spack generated host-config\n")
        cfg.write("##################################\n")
        cfg.close()

        host_cfg_fname = os.path.abspath(host_cfg_fname)
        tty.info("spack generated conduit host-config file: " + host_cfg_fname)
        return host_cfg_fname

    def cmake_args(self):
        spec = self.spec

        options = []

        if '+openmp' in spec:
            options.extend([
                '-DENABLE_OPENMP=On'])

        if '+cuda' in spec:
            options.extend([
                '-DENABLE_CUDA=On',
                '-DCUDA_TOOLKIT_ROOT_DIR=%s' % (spec['cuda'].prefix)])
        else:
            options.extend(['-DENABLE_CUDA=OFF'])

        options.extend(['-DRAJA_DIR=%s' % (spec['raja'].prefix)])
        options.extend(['-DMFEM_DIR=%s' % (spec['mfem'].prefix)])
        options.extend(['-DUMPIRE_DIR=%s' % (spec['umpire'].prefix)])
        options.extend(['-DCONDUIT_DIR=%s' % (spec['conduit'].prefix)])
        options.extend(['-DDRAY_ENABLE_TESTS=OFF'])
        options.extend(['-DENABLE_LOGGING=OFF'])
        options.extend(['-DENABLE_STATS=OFF'])
        options.extend(['../src'])

        return options
