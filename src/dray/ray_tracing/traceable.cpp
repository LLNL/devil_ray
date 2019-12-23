// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
#include <dray/ray_tracing/traceable.hpp>
#include <dray/dispatcher.hpp>

#include <dray/utils/data_logger.hpp>

#include <dray/GridFunction/device_mesh.hpp>
#include <dray/GridFunction/device_field.hpp>

namespace dray
{
namespace ray_tracing
{
namespace detail
{

// --------------------------------------------------------------------
// MatInvHack: Don't call inverse() if we can't, but get past compiler.
// --------------------------------------------------------------------
template <typename T, int32 M, int32 N>
struct MatInvHack
{
  DRAY_EXEC static Matrix<T,N,M>
  get_inverse(const Matrix<T,M,N> &m, bool &valid)
  {
    Matrix<T,N,M> a;
    a.identity();
    valid = false;
    return a;
  }
};

// ------------------------------------------------------------------------
template <typename T, int32 S>
struct MatInvHack<T, S, S>
{
  DRAY_EXEC static Matrix<T,S,S>
  get_inverse(const Matrix<T,S,S> &m, bool &valid)
  {
    return matrix_inverse(m, valid);
  }
};

// ------------------------------------------------------------------------
template <class MeshElem, class FieldElem>
Array<Fragment>
get_fragments(Mesh<MeshElem> &mesh,
              Field<FieldElem> &field,
              Array<RayHit> &hits)
{

  // Convention: If dim==2, use surface normal as direction.
  //             If dim==3, use field gradient as direction.

  const int32 size = hits.size();
  constexpr int32 dim = MeshElem::get_dim();

  //const int32 size_active_rays = rays.m_active_rays.size();

  Array<Fragment> fragments;
  fragments.resize(size);
  Fragment *fragments_ptr = fragments.get_device_ptr();

  // Initialize other outputs to well-defined dummy values.
  constexpr Vec<Float,3> one_two_three = {123., 123., 123.};


  const RayHit *hit_ptr = hits.get_device_ptr_const();

  DeviceMesh<MeshElem> device_mesh(mesh);
  DeviceField<FieldElem> device_field(field);

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, size), [=] DRAY_LAMBDA (int32 i)
  {
    Fragment frag;
    frag.m_scalar = -1.f;
    frag.m_normal = {-1.f, -1.f, -1.f};

    const RayHit &hit = hit_ptr[i];

    if(hit.m_hit_idx > -1)
    {
      const int32 el_id = hit.m_hit_idx;
      Vec<Float, dim> ref_pt;
      ref_pt[0] = hit.m_ref_pt[0];
      ref_pt[1] = hit.m_ref_pt[1];
      if(dim == 3)
      {
        ref_pt[2] = hit.m_ref_pt[2];
      }
      // Evaluate element transformation and scalar field.
      Vec<Vec<Float, 3>, dim> jac_vec;
      Vec<Float, 3> world_pos = device_mesh.get_elem(el_id).eval_d(ref_pt, jac_vec);

      Vec<Float, 1> field_val;
      Vec<Vec<Float, 1>, dim> field_deriv;  // Only init'd if dim==3.

      if (dim == 2)
        frag.m_scalar = device_field.get_elem(el_id).eval_d(ref_pt, field_deriv)[0];
      else if (dim == 3)
        frag.m_scalar = device_field.get_elem(el_id).eval_d(ref_pt, field_deriv)[0];

      // What we output as the normal depends if dim==2 or 3.
      if (dim == 2)
      {
        // Use the normalized cross product of the jacobian
        frag.m_normal = cross(jac_vec[0], jac_vec[1]);
      }
      else if (dim == 3)
      {
        // Use the gradient of the scalar field relative to world axes.
        Matrix<Float, 3, dim> jacobian_matrix;
        Matrix<Float, 1, dim> gradient_ref;
        for (int32 rdim = 0; rdim < 3; rdim++)
        {
          jacobian_matrix.set_col(rdim, jac_vec[rdim]);
          gradient_ref.set_col(rdim, field_deriv[rdim]);
        }

        // To convert to world coords, use g = gh * J_inv.
        bool inv_valid;
        const Matrix<Float, dim, 3> j_inv =
            MatInvHack<Float, 3, dim>::get_inverse(jacobian_matrix, inv_valid);
        //TODO How to handle the case that inv_valid == false?
        const Matrix<Float, 1, 3> gradient_mat = gradient_ref * j_inv;
        Vec<Float,3> gradient_world = gradient_mat.get_row(0);

        // Output.
        frag.m_normal = gradient_world;
        //TODO What if the gradient is (0,0,0)? (Matt: it will be bad)
      }
    }

    fragments_ptr[i] = frag;

  });

  return fragments;
}
} // namespace detail

// ------------------------------------------------------------------------
Traceable::Traceable(DataSet &data_set)
  : m_data_set(data_set)
{
}

// ------------------------------------------------------------------------
Traceable::~Traceable()
{
}

// ------------------------------------------------------------------------
void Traceable::input(DataSet &data_set)
{
  m_data_set = data_set;
}

// ------------------------------------------------------------------------
void Traceable::field(const std::string &field_name)
{
  m_field_name = field_name;
}

// ------------------------------------------------------------------------
void Traceable::color_map(ColorMap &color_map)
{
  m_color_map = color_map;
}

// ------------------------------------------------------------------------
ColorMap& Traceable::color_map()
{
  return m_color_map;
}

// ------------------------------------------------------------------------
struct FragmentFunctor
{
  Array<RayHit> *m_hits;
  Array<Fragment> m_fragments;
  FragmentFunctor(Array<RayHit> *hits)
    : m_hits(hits)
  {
  }

  template<typename TopologyType, typename FieldType>
  void operator()(TopologyType &topo, FieldType &field)
  {
    m_fragments = detail::get_fragments(topo.mesh(), field, *m_hits);
  }
};

// ------------------------------------------------------------------------
Array<Fragment>
Traceable::fragments(Array<RayHit> &hits)
{
  DRAY_LOG_OPEN("fragments");
  assert(m_field_name != "");

  TopologyBase *topo = m_data_set.topology();
  FieldBase *field = m_data_set.field(m_field_name);

  FragmentFunctor func(&hits);
  dispatch_3d(topo, field, func);
  DRAY_LOG_CLOSE();
  return func.m_fragments;
}


}} // namespace dray::ray_tracing
