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


  template <int32 ncomp>
  DRAY_EXEC void NewtonBernstein1D(const int32 p,
                                   const Float *x,
                                   Float *w,
                                   Vec<Float, ncomp> *f,
                                   Vec<Float, ncomp> *c)
  {
    c[0] = f[0];
    w[0] = 1;
    for (int32 k = 1; k <= p; ++k)
    {
      for (int32 j = p; j >= k; --j)
        f[j] = (f[j]-f[j-1]) / (x[j] - x[j-k]);
      for (int32 j = k; j >= 1; --j)
      {
        w[j] = w[j-1] * ((1.0f*j/k) * (1-x[k-1])) -  w[j] * ((1-1.0f*j/k) * x[k-1]);
        c[j] = c[j-1] * (1.0f*j/k) + c[j] * (1-1.0f*j/k) + f[k] * w[j];
      }
      w[0] = -w[0] * x[k-1];
      c[0] = c[0] + f[k] * w[0];
    }
  }




  /**
   * @brief Solves Bernstein interpolation problem on each element.
   *
   * TODO Handle shared degrees of freedom.
   * TODO Handle non-uniformly spaced points.
   */

  template <int32 ncomp>
  GridFunction<ncomp> ToBernstein_execute(const ShapeHex,
                                          const GridFunction<ncomp> &in,
                                          const int32 p_)
  {
    GridFunction<ncomp> out;
    out.resize(in.m_size_el, in.m_el_dofs, in.m_size_ctrl);
    array_copy(out.m_ctrl_idx, in.m_ctrl_idx);

    DeviceGridFunctionConst<ncomp> dgf_in(in);
    DeviceGridFunction<ncomp> dgf_out(out);
    const int32 nelem = in.m_size_el;
    const int32 p = p_;
    const OrderPolicy<General> order_p{p};

    // Convert each element.
    RAJA::forall<for_policy>(RAJA::RangeSegment(0, nelem), [=] DRAY_LAMBDA (int32 eidx) {
        ReadDofPtr<Vec<Float, ncomp>> rdp = dgf_in.get_rdp(eidx);
        WriteDofPtr<Vec<Float, ncomp>> wdp = dgf_out.get_wdp(eidx);

        // 1D scratch space.
        //   TODO shrink this when get refined General policy.
        Vec<Float, ncomp> bF[MaxPolyOrder+1];
        Vec<Float, ncomp> bC[MaxPolyOrder+1];
        Float bW[MaxPolyOrder+1];

        // Uniform closed.  TODO more general spacing options.
        Float x[MaxPolyOrder+1];
        for (int32 i = 0; i <= p; ++i)
          x[i] = 1.0 * i / p;

        const int32 npe = (p+1)*(p+1)*(p+1);
        for (int32 nidx = 0; nidx < npe; ++nidx)
          wdp[nidx] = rdp[nidx];

#if 0
        // i
        for (int32 k = 0; k <= p; ++k)
          for (int32 j = 0; j <= p; ++j)
          {
            for (int32 i = 0; i <= p; ++i)
              bF[i] = wdp[k*(p+1)*(p+1) + j*(p+1) + i];
            NewtonBernstein1D(p, x, bW, bF, bC);
            for (int32 i = 0; i <= p; ++i)
              wdp[k*(p+1)*(p+1) + j*(p+1) + i] = bC[i];
          }
#endif

#if 1
        // j
        for (int32 k = 0; k <= p; ++k)
          for (int32 i = 0; i <= p; ++i)
          {
            for (int32 j = 0; j <= p; ++j)
              bF[j] = wdp[k*(p+1)*(p+1) + j*(p+1) + i];
            NewtonBernstein1D(p, x, bW, bF, bC);
            for (int32 j = 0; j <= p; ++j)
              wdp[k*(p+1)*(p+1) + j*(p+1) + i] = bC[j];
          }
#endif

#if 0
        // k
        for (int32 j = 0; j <= p; ++j)
          for (int32 i = 0; i <= p; ++i)
          {
            for (int32 k = 0; k <= p; ++k)
              bF[k] = wdp[k*(p+1)*(p+1) + j*(p+1) + i];
            NewtonBernstein1D(p, x, bW, bF, bC);
            for (int32 k = 0; k <= p; ++k)
              wdp[k*(p+1)*(p+1) + j*(p+1) + i] = bC[k];
          }
#endif
    });

    return out;
  }

  template <int32 ncomp>
  GridFunction<ncomp> ToBernstein_execute(const ShapeQuad,
                                          const GridFunction<ncomp> &in,
                                          const int32 p_)

  {
    throw std::logic_error("ToBernstein_execute(ShapeQuad, gf) not implemented");
  }

  template <int32 ncomp>
  GridFunction<ncomp> ToBernstein_execute(const ShapeTet,
                                          const GridFunction<ncomp> &in,
                                          const int32 p_)

  {
    throw std::logic_error("ToBernstein_execute(ShapeTet, gf) not implemented");
  }

  template <int32 ncomp>
  GridFunction<ncomp> ToBernstein_execute(const ShapeTri,
                                          const GridFunction<ncomp> &in,
                                          const int32 p_)

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
        ToBernstein_execute(adapt_get_shape<MElemT>(), in_mesh_gf, topo.order());
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
        ToBernstein_execute(adapt_get_shape<FElemT>(), in_gf, field.order());

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
