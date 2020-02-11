// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_MFEM_HEADERS_HPP
#define DRAY_MFEM_HEADERS_HPP

// get the min required headers from MFEM
// this lets us know about mfem without using MPI,
// which is needed when mfem is built with MPI
#include <mfem/fem/intrules.hpp>
#include <mfem/fem/geom.hpp>
#include <mfem/fem/fe.hpp>
#include <mfem/fem/fe_coll.hpp>
#include <mfem/fem/eltrans.hpp>
#include <mfem/fem/coefficient.hpp>
#include <mfem/fem/lininteg.hpp>
#include <mfem/fem/nonlininteg.hpp>
#include <mfem/fem/bilininteg.hpp>
#include <mfem/fem/fespace.hpp>
#include <mfem/fem/gridfunc.hpp>
#include <mfem/fem/linearform.hpp>
#include <mfem/fem/nonlinearform.hpp>
#include <mfem/fem/bilinearform.hpp>
#include <mfem/fem/hybridization.hpp>
#include <mfem/fem/datacollection.hpp>
#include <mfem/fem/estimators.hpp>
#include <mfem/fem/staticcond.hpp>
//
//#include <mfem/mesh/mesh_headers.hpp>
#include <mfem/mesh/vertex.hpp>
#include <mfem/mesh/element.hpp>
#include <mfem/mesh/point.hpp>
#include <mfem/mesh/segment.hpp>
#include <mfem/mesh/triangle.hpp>
#include <mfem/mesh/quadrilateral.hpp>
#include <mfem/mesh/hexahedron.hpp>
#include <mfem/mesh/tetrahedron.hpp>
#include <mfem/mesh/ncmesh.hpp>
#include <mfem/mesh/mesh.hpp>
#include <mfem/mesh/mesh_operators.hpp>
#include <mfem/mesh/nurbs.hpp>
#include <mfem/mesh/wedge.hpp>

#endif
