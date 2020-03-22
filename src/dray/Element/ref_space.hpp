// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_REF_SPACE
#define DRAY_REF_SPACE

#include <dray/types.hpp>
#include <dray/exports.hpp>
#include <dray/Element/elem_attr.hpp>
#include <dray/Element/subref.hpp>

namespace dray
{

template <int32 dim, ElemType etype>
struct RefSpaceTag {};

// Note : as long as the "function overload only" version works,
//        no need for the class template specialization workaround.
//
///   namespace specials  // Workarounds for specializing function templates.
///   {
///     template <class RefSpaceTagT>
///     struct ref_universe_struct { };
/// 
///     // ref_universe<Tri>
///     template <int32 dim>
///     struct ref_universe_struct<RefSpaceTag<dim, ElemType::Tri>>
///     {
///       typedef SubRef<dim, ElemType::Tri> ret_type;
/// 
///       DRAY_EXEC static SubRef<dim, ElemType::Tri> f()
///       {
///         SubRef<dim, ElemType::Tri> subtri;
///         for (int32 d = 0; d < dim; ++d)
///         {
///           subtri[d] = 0.0f;
///           subtri[d][d] = 1.0f;  // standard basis axes
///         }
///         subtri[dim] = 0.0f;  // origin
///         return subtri;
///       }
///     };
/// 
///     // ref_universe<Quad>
///     template <int32 dim>
///     struct ref_universe_struct<RefSpaceTag<dim, ElemType::Quad>>
///     {
///       typedef SubRef<dim, ElemType::Quad> ret_type;
/// 
///       DRAY_EXEC static SubRef<dim, ElemType::Quad> f()
///       {
///         SubRef<dim, ElemType::Quad> subcube;
///         for (int32 d = 0; d < dim; ++d)
///         {
///           subcube.m_ranges[d].include(0.0f);
///           subcube.m_ranges[d].include(1.0f);   // unit interval
///         }
///         return subcube;
///       }
///     };
///   }
///
/// //
/// // ref_universe<>()
/// //
/// template <class RefSpaceTagT>
/// DRAY_EXEC auto ref_universe(const RefSpaceTagT) -> typename specials::ref_universe_struct<RefSpaceTagT>::ret_type
/// {
///   return specials::ref_universe_struct<RefSpaceTagT>::f();
/// }

// --------------------------------------------------------------------
// ref_universe()
// --------------------------------------------------------------------

//
// ref_universe<Tri>()
//
template <int32 dim>
DRAY_EXEC SubRef<dim, ElemType::Tri>
ref_universe(const RefSpaceTag<dim, ElemType::Tri>)
{
  SubRef<dim, ElemType::Tri> subtri;
  for (int32 d = 0; d < dim; ++d)
  {
    subtri[d] = 0.0f;
    subtri[d][d] = 1.0f;  // standard basis axes
  }
  subtri[dim] = 0.0f;  // origin
  return subtri;
}

//
// ref_universe<Quad>()
//
template <int32 dim>
DRAY_EXEC SubRef<dim, ElemType::Quad>
ref_universe(const RefSpaceTag<dim, ElemType::Quad>)
{
  SubRef<dim, ElemType::Quad> subcube;
  subcube[0] = 0;
  subcube[1] = 1;
  /// for (int32 d = 0; d < dim; ++d)
  /// {
  ///   subcube.m_ranges[d].include(0.0f);
  ///   subcube.m_ranges[d].include(1.0f);   // unit interval
  /// }
  return subcube;
}
// --------------------------------------------------------------------



// --------------------------------------------------------------------
// split_subref()
// --------------------------------------------------------------------

//
// split_subref<Tri>
//
template <int32 dim>
DRAY_EXEC SubRef<dim, ElemType::Tri>
split_subref(const SubRef<dim, ElemType::Tri> &subref, const Split<ElemType::Tri> &sp)
{
  SubRef<dim, ElemType::Tri> tri{subref};
  tri[sp.vtx_displaced] = (tri[sp.vtx_tradeoff] * sp.factor) +
                          (tri[sp.vtx_displaced] * (1.0f - sp.factor));
  return tri;
}

//
// split_subref<Quad>
//
template <int32 dim>
DRAY_EXEC SubRef<dim, ElemType::Quad>
split_subref(const SubRef<dim, ElemType::Quad> &subref, const Split<ElemType::Quad> &sp)
{
  SubRef<dim, ElemType::Quad> cube{subref};
  const Float split_point = (cube[0][sp.axis] * (1.0f - sp.factor)) +
                            (cube[1][sp.axis] * sp.factor);
  if (!sp.f_lower_t_upper)
    cube[1][sp.axis] = split_point;
  else
    cube[0][sp.axis] = split_point;

  return cube;


  /// SubRef<dim, ElemType::Quad> cube{subref};

  /// const Float v0 = cube.m_ranges[sp.axis].min();
  /// const Float v1 = cube.m_ranges[sp.axis].max();

  /// cube.m_ranges[sp.axis].reset();

  /// const Float split_point = (v0 * (1.0f - sp.factor)) +
  ///                           (v1 * sp.factor);

  /// cube.m_ranges[sp.axis].include(split_point);

  /// if (!sp.f_lower_t_upper)
  ///   cube.m_ranges[sp.axis].include(v0);
  /// else
  ///   cube.m_ranges[sp.axis].include(v1);

  /// return cube;
}
// --------------------------------------------------------------------



// Temporarily relocated QuadRefSpace, TriRefSpace, RefTri here
// until Element no longer depends on them.

template <int32 dim>
class QuadRefSpace
{
  public:
  DRAY_EXEC static bool is_inside (const Vec<Float, dim> &ref_coords); // TODO
  DRAY_EXEC static bool is_inside (const Vec<Float, dim> &ref_coords,
                                   const Float &eps);
  DRAY_EXEC static void clamp_to_domain (Vec<Float, dim> &ref_coords); // TODO
  DRAY_EXEC static Vec<Float, dim>
  project_to_domain (const Vec<Float, dim> &r1, const Vec<Float, dim> &r2); // TODO
};

template <int32 dim>
class TriRefSpace
{
  public:
  DRAY_EXEC static bool is_inside (const Vec<Float, dim> &ref_coords); // TODO
  DRAY_EXEC static bool is_inside (const Vec<Float, dim> &ref_coords,
                                   const Float &eps);
  DRAY_EXEC static void clamp_to_domain (Vec<Float, dim> &ref_coords); // TODO
  DRAY_EXEC static Vec<Float, dim>
  project_to_domain (const Vec<Float, dim> &r1, const Vec<Float, dim> &r2); // TODO
};

template <int32 dim> struct RefTri
{
  Vec<float32, dim> m_vertices[dim + 1];

  DRAY_EXEC static RefTri ref_universe ()
  {
    RefTri ret;
    for (int d = 0; d < dim; d++)
    {
      ret.m_vertices[d] = 0.0f;
      ret.m_vertices[d][d] = 1.0f;
      ret.m_vertices[dim][d] = 1.0f;
    }
    return ret;
  }

  DRAY_EXEC Vec<float32, dim> center () const
  {
    Vec<float32, dim> c;
    c = 0.0;
    for (int d = 0; d <= dim; d++)
      c += m_vertices[d];
    c *= float32 (1.0 / (dim + 1));
    return c;
  }

  DRAY_EXEC float32 max_length () const
  {
    // Any proxy for diameter. In this case use maximum edge length.
    float32 M = 0.0;
    for (int32 v1 = 0; v1 <= dim; v1++)
      for (int32 v2 = 0; v2 <= dim; v2++)
        M = fmaxf (M, (m_vertices[v1] - m_vertices[v2]).magnitude2 ());
    return sqrtf (M);
  }
};


}//namespace dray

#endif//DRAY_REF_SPACE
