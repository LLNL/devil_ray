// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <dray/filters/to_bernstein.hpp>
#include <dray/error.hpp>
#include <dray/dispatcher.hpp>
#include <dray/array_utils.hpp>

#include <RAJA/RAJA.hpp>
#include <dray/policies.hpp>
#include <dray/exports.hpp>

#include <dray/topology_base.hpp>
#include <dray/derived_topology.hpp>
#include <dray/GridFunction/mesh.hpp>
#include <dray/GridFunction/field.hpp>
#include <dray/GridFunction/device_mesh.hpp>
#include <dray/GridFunction/device_field.hpp>
#include <dray/Element/elem_attr.hpp>


/**
 * Algorithms based on (Ainsworth & Sanchez, 2016).
 *
 * @article{ainsworth2016computing,
 *   title={Computing the Bezier control points of the Lagrangian interpolant in arbitrary dimension},
 *   author={Ainsworth, Mark and S{\'a}nchez, Manuel A},
 *   journal={SIAM Journal on Scientific Computing},
 *   volume={38},
 *   number={3},
 *   pages={A1682--A1700},
 *   year={2016},
 *   publisher={SIAM}
 * }
 */

namespace dray
{

  // Dispatch topology
  // Create new dataset
  //
  // For each field,
  //   dispatch field
  //   add to dataset
  //
  //
  // Task:
  //   (Assume that no dofs are shared)
  //   Make identically shaped grid function.
  //   RAJA-for each element,
  //     Create ReadDofPtr from input gf, WriteDofPtr from output gf.
  //     1D buffers
  //     Work magic


  // ---------- Implementation ------------------- //

  /**
   * @brief Solves Bernstein interpolation problem on each element.
   *
   * TODO Handle shared degrees of freedom.
   * TODO Handle non-uniformly spaced points.
   */

  template <int32 ncomp>
  GridFunction<ncomp> ToBernstein_execute(const ShapeHex, const GridFunction<ncomp> &in)
  {
    GridFunction<ncomp> out;
    out.resize(in.m_size_el, in.m_el_dofs, in.m_size_ctrl);

    //TODO
    //STUB
    array_copy(out.m_ctrl_idx, in.m_ctrl_idx);
    array_copy(out.m_values, in.m_values);

    return out;
  }

  template <int32 ncomp>
  GridFunction<ncomp> ToBernstein_execute(const ShapeQuad, const GridFunction<ncomp> &in)
  {
    throw std::logic_error("ToBernstein_execute(ShapeQuad, gf) not implemented");
  }

  template <int32 ncomp>
  GridFunction<ncomp> ToBernstein_execute(const ShapeTet, const GridFunction<ncomp> &in)
  {
    throw std::logic_error("ToBernstein_execute(ShapeTet, gf) not implemented");
  }

  template <int32 ncomp>
  GridFunction<ncomp> ToBernstein_execute(const ShapeTri, const GridFunction<ncomp> &in)
  {
    throw std::logic_error("ToBernstein_execute(ShapeTri, gf) not implemented");
  }


  // ---------- Wrappers ------------------- //

  // ToBernsteinTopo_execute(): Get grid function and pass to ToBernstein_execute().
  template <typename MElemT>
  std::shared_ptr<TopologyBase> ToBernsteinTopo_execute(
      const DerivedTopology<MElemT> &topo)
  {
    const GridFunction<3> &in_mesh_gf = topo.mesh().get_dof_data();
    const GridFunction<3> out_mesh_gf =
        ToBernstein_execute(adapt_get_shape<MElemT>(), in_mesh_gf);
    Mesh<MElemT> mesh(out_mesh_gf, topo.order());

    return std::make_shared<DerivedTopology<MElemT>>(mesh);
  }

  // ToBernsteinField_execute(): Get grid function and pass to ToBernstein_execute().
  template <typename FElemT>
  std::shared_ptr<FieldBase> ToBernsteinField_execute(const Field<FElemT> &field)
  {
    constexpr int32 ncomp = FElemT::get_ncomp();
    const GridFunction<ncomp> &in_gf = field.get_dof_data();
    const GridFunction<ncomp> out_gf =
        ToBernstein_execute(adapt_get_shape<FElemT>(), in_gf);

    return std::make_shared<Field<FElemT>>(out_gf, field.order(), field.name());
  }


  // Templated topology functor
  struct ToBernstein_TopoFunctor
  {
    std::shared_ptr<TopologyBase> m_output;

    template <typename TopologyT>
    void operator() (TopologyT &topo)
    {
      m_output = ToBernsteinTopo_execute(topo);
    }
  };

  // Templated field functor
  struct ToBernstein_FieldFunctor
  {
    std::shared_ptr<FieldBase> m_output;

    template <typename FieldT>
    void operator() (FieldT &field)
    {
      m_output = ToBernsteinField_execute(field);
    }
  };

  // execute() wrapper
  DataSet ToBernstein::execute(DataSet &data_set)
  {
    ToBernstein_TopoFunctor topo_f;
    ToBernstein_FieldFunctor field_f;

    dispatch(data_set.topology(), topo_f);
    DataSet out_ds(topo_f.m_output);

    for (const std::string &fname : data_set.fields())
    {
      dispatch(data_set.field(fname), field_f);
      out_ds.add_field(field_f.m_output);
    }

    return out_ds;
  }

}//namespace dray
