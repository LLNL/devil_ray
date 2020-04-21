// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_ISO_OPS_HPP
#define DRAY_ISO_OPS_HPP

#include <dray/types.hpp>
#include <dray/math.hpp>
#include <dray/Element/element.hpp>
#include <dray/Element/dof_access.hpp>
#include <dray/Element/elem_ops.hpp>
#include <dray/Element/subref.hpp>

namespace dray
{


// ----------------------------------------------------
// Isosurfacing approach based on
//   https://dx.doi.org/10.1016/j.cma.2016.10.019
//
// @article{FRIES2017759,
//   title = "Higher-order meshing of implicit geometries—Part I: Integration and interpolation in cut elements",
//   journal = "Computer Methods in Applied Mechanics and Engineering",
//   volume = "313",
//   pages = "759 - 784",
//   year = "2017",
//   issn = "0045-7825",
//   doi = "https://doi.org/10.1016/j.cma.2016.10.019",
//   url = "http://www.sciencedirect.com/science/article/pii/S0045782516308696",
//   author = "T.P. Fries and S. Omerović and D. Schöllhammer and J. Steidl",
// }
// ----------------------------------------------------

  namespace eops
  {

  /**
   * RotatedIdx
   *
   *   @brief An multi-index + a permutation for local ordering + linearizer.
   *          Purpose: iterate over faces/edges of a 3D element.
   *
   *   ApparentCoefficient[i,j] = dofs[ oriented_idx.linearize(i,j) ]
   *                            == dofs[ m_linearizer( {indices with i applied to axis pi[0],
   *                                                                 j applied to axis pi[1],...}) ];
   */
  // RotatedIdx3, for hex
  template <int8 pi0, int8 pi1, int8 pi2, typename LinearizerT>
  struct RotatedIdx3
  {
    protected:
      mutable Vec<int32, 3> m_I;
      const LinearizerT m_linearizer;

      DRAY_EXEC void apply(int32 i)                   const { m_I[pi0]+=i; }
      DRAY_EXEC void apply(int32 i, int32 j)          const { m_I[pi0]+=i; m_I[pi1]+=j; }
      DRAY_EXEC void apply(int32 i, int32 j, int32 k) const { m_I[pi0]+=i; m_I[pi1]+=j; m_I[pi2]+=k; }

    public:
      DRAY_EXEC RotatedIdx3(int32 start0, int32 start1, int32 start2, const LinearizerT & linearizer)
      : m_I{{start0, start1, start2}},
        m_linearizer{linearizer}
      { }

      DRAY_EXEC int32 linearize(int32 i) const
      {
        apply(i);
        const int32 idx = m_linearizer(m_I[0], m_I[1], m_I[2]);
        apply(-i);
        return idx;
      }

      DRAY_EXEC int32 linearize(int32 i, int32 j) const
      {
        apply(i, j);
        const int32 idx = m_linearizer(m_I[0], m_I[1], m_I[2]);
        apply(-i, -j);
        return idx;
      }

      DRAY_EXEC int32 linearize(int32 i, int32 j, int32 k) const
      {
        apply(i, j, k);
        const int32 idx = m_linearizer(m_I[0], m_I[1], m_I[2]);
        apply(-i, -j, -k);
        return idx;
      }
  };

  // RotatedIdx4, for tet
  template <int8 pi0, int8 pi1, int8 pi2, int8 pi3, typename LinearizerT>
  struct RotatedIdx4
  {
    protected:
      mutable Vec<int32, 4> m_I;
      const LinearizerT m_linearizer;

      DRAY_EXEC void apply(int32 i)                            const { m_I[pi0]+=i; }
      DRAY_EXEC void apply(int32 i, int32 j)                   const { m_I[pi0]+=i; m_I[pi1]+=j; }
      DRAY_EXEC void apply(int32 i, int32 j, int32 k)          const { m_I[pi0]+=i; m_I[pi1]+=j; m_I[pi2]+=k; }
      DRAY_EXEC void apply(int32 i, int32 j, int32 k, int32 l) const { m_I[pi0]+=i; m_I[pi1]+=j; m_I[pi2]+=k; m_I[pi3]+=l; }

    public:
      DRAY_EXEC RotatedIdx4(int32 start0, int32 start1, int32 start2, int32 start3, const LinearizerT & linearizer)
      : m_I{{start0, start1, start2, start3}},
        m_linearizer{linearizer}
      { }

      DRAY_EXEC int32 linearize(int32 i) const
      {
        apply(i);
        const int32 idx = m_linearizer(m_I[0], m_I[1], m_I[2], m_I[3]);
        apply(-i);
        return idx;
      }

      DRAY_EXEC int32 linearize(int32 i, int32 j) const
      {
        apply(i, j);
        const int32 idx = m_linearizer(m_I[0], m_I[1], m_I[2], m_I[3]);
        apply(-i, -j);
        return idx;
      }

      DRAY_EXEC int32 linearize(int32 i, int32 j, int32 k) const
      {
        apply(i, j, k);
        const int32 idx = m_linearizer(m_I[0], m_I[1], m_I[2], m_I[3]);
        apply(-i, -j, -k);
        return idx;
      }

      DRAY_EXEC int32 linearize(int32 i, int32 j, int32 k, int32 l) const
      {
        apply(i, j, k, l);
        const int32 idx = m_linearizer(m_I[0], m_I[1], m_I[2], m_I[3]);
        apply(-i, -j, -k, -l);
        return idx;
      }
  };


  struct HexFlat
  {
    int32 m_order;
    int32 operator()(int32 i, int32 j, int32 k) const
    {
      return i + (m_order+1)*j + (m_order+1)*(m_order+1)*k;
    }
  };

  struct TetFlat
  {
    int32 m_order;
    int32 operator()(int32 i, int32 j, int32 k, int32 mu) const
    {
      return detail::cartesian_to_tet_idx(i, j, k, (m_order+1));
    }
  };


  //
  // Section 3.1 Valid level-set data and recursion.
  //   - Criteria defining whether further splits are needed
  //     before the isocut is considered 'simple'.
  //
  //   - The authors use a sampling approach to determine whether
  //     an element/face/edge is cut.
  //
  //   - We can use conservative Bernstein estimates instead.
  //
  //   2D:
  //     (i) Each element edge is only cut once;  |  Bounded by #(+/-) changes
  //                                              |  ('variation diminishing').
  //
  //     (ii) The overall number of cut           |  Number of edges whose bounds
  //          edges must be two;                  |  contain $\iota$ must be two.
  //
  //     (iii) If no edge is cut then             |  Edge bounds contain $\iota$
  //           the element is completely uncut.   |  or element bounds don't either.
  //
  //   3D:
  //     (i)--(iii) must hold on each face;
  //
  //     (iv) If no face is cut then              |  Face bounds contain $\iota$
  //          the element is completely uncut.    |  or element bounds don't either.
  //
  //   Based on these criteria, we should define a method to check
  //   if they are all satisfied for a given element. If one or
  //   more criteria are violated, suggest an effective split
  //   to resolve the violation.


  using ScalarDP = ReadDofPtr<Vec<Float, 1>>;

  int8 isosign(Float value, Float isovalue)
  {
    return (value < isovalue - epsilon<Float>() ? -1
            : value > isovalue + epsilon<Float>() ? +1
            : 0);
  }

  // TODO use sampling because the control points will be too generous.
  template <class RotatedIndexT>
  int32 edge_var(const RotatedIndexT &wheel, const ScalarDP &dofs, Float iota, int32 p)
  {
    int32 count = 0;
    // TODO review watertight isosurfaces, what to do when equal.
    int8 prev_s = isosign(dofs[wheel.linearize(0)][0], iota);
    for (int32 i = 1; i <= p; ++i)
    {
      int8 new_s = isosign(dofs[wheel.linearize(i)][0], iota);
      if (prev_s && new_s && (new_s != prev_s))
      {
        prev_s = new_s;
        count++;
      }
    }
    return count;
  }

  template <class RotatedIndexT>
  bool face_cut_hex(const RotatedIndexT &wheel, const ScalarDP &dofs, Float iota, int32 p)
  {
    Range dof_range;
    for (int j = 0; j <=p; ++j)
      for (int i = 0; i <=p; ++i)
        dof_range.include(dofs[wheel.linearize(i,j)][0]);
    return dof_range.contains(iota);
  }

  bool int_cut_hex(const ScalarDP &dofs, Float iota, int32 p)
  {
    Range dof_range;
    const int32 ndofs = (p+1)*(p+1)*(p+1);
    for (int i = 0; i < ndofs; ++i)
      dof_range.include(dofs[i][0]);
    return dof_range.contains(iota);
  }


  namespace hex_enums
  {
    enum Edges { e00=(1u<< 0),  e01=(1u<< 1),  e02=(1u<< 2),  e03=(1u<< 3),
                 e04=(1u<< 4),  e05=(1u<< 5),  e06=(1u<< 6),  e07=(1u<< 7),
                 e08=(1u<< 8),  e09=(1u<< 9),  e10=(1u<<10),  e11=(1u<<11) };

    enum Faces { f00=(1u<<0), f01=(1u<<1), f02=(1u<<2),
                 f03=(1u<<3), f04=(1u<<4), f05=(1u<<5) };
  }


  struct IsocutInfo
  {
    enum CutOptions { Cut = 1u,         CutSimpleTri = 2u, CutSimpleQuad = 4u,
                      IntNoFace = 8u,   IntManyFace = 16u,
                      FaceNoEdge = 32u, FaceManyEdge = 64u,
                      EdgeManyPoint = 128u };
    uint8 m_cut_type_flag;
    uint8 m_bad_faces_flag;
    uint32 m_bad_edges_flag;

    void clear() { m_cut_type_flag = 0;  m_bad_faces_flag = 0;  m_bad_edges_flag = 0; }
  };

  DRAY_EXEC IsocutInfo measure_isocut(ShapeHex, const ScalarDP & dofs, Float iota, int32 p)
  {
    IsocutInfo info;
    info.clear();

    const HexFlat hlin{p};

    using namespace hex_enums;

    // All cut edges and bad edges (bad = cut more than once).
    uint32 ce = 0u;
    uint32 be = 0u;
    int32 ev;

    // X aligned edges
    ev = edge_var(RotatedIdx3<0,1,2, HexFlat>(0,0,0, hlin), dofs, iota, p);
      ce |= e00 * (ev > 0);
      be |= e00 * (ev > 1);
    ev = edge_var(RotatedIdx3<0,1,2, HexFlat>(0,p,0, hlin), dofs, iota, p);
      ce |= e01 * (ev > 0);
      be |= e01 * (ev > 1);
    ev = edge_var(RotatedIdx3<0,1,2, HexFlat>(0,0,p, hlin), dofs, iota, p);
      ce |= e02 * (ev > 0);
      be |= e02 * (ev > 1);
    ev = edge_var(RotatedIdx3<0,1,2, HexFlat>(0,p,p, hlin), dofs, iota, p);
      ce |= e03 * (ev > 0);
      be |= e03 * (ev > 1);

    // Y aligned edges
    ev = edge_var(RotatedIdx3<1,2,0, HexFlat>(0,0,0, hlin), dofs, iota, p);
      ce |= e04 * (ev > 0);
      be |= e04 * (ev > 1);
    ev = edge_var(RotatedIdx3<1,2,0, HexFlat>(p,0,0, hlin), dofs, iota, p);
      ce |= e05 * (ev > 0);
      be |= e05 * (ev > 1);
    ev = edge_var(RotatedIdx3<1,2,0, HexFlat>(0,0,p, hlin), dofs, iota, p);
      ce |= e06 * (ev > 0);
      be |= e06 * (ev > 1);
    ev = edge_var(RotatedIdx3<1,2,0, HexFlat>(p,0,p, hlin), dofs, iota, p);
      ce |= e07 * (ev > 0);
      be |= e07 * (ev > 1);

    // Z aligned edges
    ev = edge_var(RotatedIdx3<2,0,1, HexFlat>(0,0,0, hlin), dofs, iota, p);
      ce |= e08 * (ev > 0);
      be |= e08 * (ev > 1);
    ev = edge_var(RotatedIdx3<2,0,1, HexFlat>(p,0,0, hlin), dofs, iota, p);
      ce |= e09 * (ev > 0);
      be |= e09 * (ev > 1);
    ev = edge_var(RotatedIdx3<2,0,1, HexFlat>(0,p,0, hlin), dofs, iota, p);
      ce |= e10 * (ev > 0);
      be |= e10 * (ev > 1);
    ev = edge_var(RotatedIdx3<2,0,1, HexFlat>(p,p,0, hlin), dofs, iota, p);
      ce |= e11 * (ev > 0);
      be |= e11 * (ev > 1);

    // Update info with edges.
    info.m_bad_edges_flag = be;
    info.m_cut_type_flag |= IsocutInfo::EdgeManyPoint * bool(be);

    // All cut faces.
    uint8 cf = 0;
    cf |= f00 * face_cut_hex(RotatedIdx3<0,1,2, HexFlat>(0,0,0, hlin), dofs, iota, p);
    cf |= f01 * face_cut_hex(RotatedIdx3<0,1,2, HexFlat>(0,0,p, hlin), dofs, iota, p);
    cf |= f02 * face_cut_hex(RotatedIdx3<1,2,0, HexFlat>(0,0,0, hlin), dofs, iota, p);
    cf |= f03 * face_cut_hex(RotatedIdx3<1,2,0, HexFlat>(p,0,0, hlin), dofs, iota, p);
    cf |= f04 * face_cut_hex(RotatedIdx3<2,0,1, HexFlat>(0,0,0, hlin), dofs, iota, p);
    cf |= f05 * face_cut_hex(RotatedIdx3<2,0,1, HexFlat>(0,p,0, hlin), dofs, iota, p);

    // FaceNoEdge (A face that is cut without any of its edges being cut).
    uint8 fne = 0;
    fne |= f00 * ((cf & f00) && !(ce & (e00 | e01 | e04 | e05)));
    fne |= f01 * ((cf & f01) && !(ce & (e02 | e03 | e06 | e07)));
    fne |= f02 * ((cf & f02) && !(ce & (e04 | e06 | e08 | e10)));
    fne |= f03 * ((cf & f03) && !(ce & (e05 | e07 | e09 | e11)));
    fne |= f04 * ((cf & f04) && !(ce & (e00 | e02 | e08 | e09)));
    fne |= f05 * ((cf & f05) && !(ce & (e01 | e03 | e10 | e11)));

    // FaceManyEdge (A face for which more than two incident edges are cut).
    uint8 fme = 0;
    fme |= f00 * (bool(ce & e00) + bool(ce & e01) + bool(ce & e04) + bool(ce & e05) > 2);
    fme |= f01 * (bool(ce & e02) + bool(ce & e03) + bool(ce & e06) + bool(ce & e07) > 2);
    fme |= f02 * (bool(ce & e04) + bool(ce & e06) + bool(ce & e08) + bool(ce & e10) > 2);
    fme |= f03 * (bool(ce & e05) + bool(ce & e07) + bool(ce & e09) + bool(ce & e11) > 2);
    fme |= f04 * (bool(ce & e00) + bool(ce & e02) + bool(ce & e08) + bool(ce & e09) > 2);
    fme |= f05 * (bool(ce & e01) + bool(ce & e03) + bool(ce & e10) + bool(ce & e11) > 2);

    // Update info with faces.
    info.m_bad_faces_flag |= fne | fme;
    info.m_cut_type_flag |= IsocutInfo::FaceNoEdge * bool(fne);
    info.m_cut_type_flag |= IsocutInfo::FaceManyEdge * bool(fme);

    const int8 num_cut_faces
      = (uint8(0) + bool(cf & f00) + bool(cf & f01) + bool(cf & f02)
                  + bool(cf & f03) + bool(cf & f04) + bool(cf & f05));

    const bool ci = int_cut_hex(dofs, iota, p);

    // Update info with interior.
    info.m_cut_type_flag |= IsocutInfo::IntNoFace * (ci && !cf);
    info.m_cut_type_flag |= IsocutInfo::IntManyFace * (num_cut_faces > 4);

    // Cut or not.
    info.m_cut_type_flag |= IsocutInfo::Cut * (ci || cf || ce);

    // Combine all info to describe whether the cut is simple.
    if (info.m_cut_type_flag < 8)
    {
      info.m_cut_type_flag |= IsocutInfo::CutSimpleTri *  (num_cut_faces == 3);
      info.m_cut_type_flag |= IsocutInfo::CutSimpleQuad * (num_cut_faces == 4);
    }

    return info;
  }


  Split<Tensor> pick_iso_simple_split(ShapeHex, const IsocutInfo &info)
  {
    using namespace hex_enums;

    const uint8 &bf = info.m_bad_faces_flag;
    const uint32 &be = info.m_bad_edges_flag;

    Float score_x = 0, score_y = 0, score_z = 0;

    // Problematic edges on an axis increase likelihood that axis is split.
    score_x += 0.25f * (bool(be & e00) + bool(be & e01) + bool(be & e02) + bool(be & e03));
    score_y += 0.25f * (bool(be & e04) + bool(be & e05) + bool(be & e06) + bool(be & e07));
    score_z += 0.25f * (bool(be & e08) + bool(be & e09) + bool(be & e10) + bool(be & e11));

    // Problematic faces normal to an axis decrease likelihood that axis is split.
    score_x -= 0.5f * (bool(bf & f02) + bool(bf & f03));
    score_y -= 0.5f * (bool(bf & f04) + bool(bf & f05));
    score_z -= 0.5f * (bool(bf & f00) + bool(bf & f01));

    const int32 split_axis = (score_x > score_y ?
                               (score_x > score_z ? 0 : 2) :
                               (score_y > score_z ? 1 : 2));

    return Split<Tensor>::half(split_axis);
  }




  DRAY_EXEC IsocutInfo measure_isocut(ShapeTet, const ScalarDP & dofs, Float iota, int32 p)
  {
    std::cerr << "Bad " << __FILE__ << "  " << __LINE__ << "\n";
    return *(IsocutInfo*)nullptr;
  }


  Split<Simplex> pick_iso_simple_split(ShapeTet, const IsocutInfo &info)
  {
    std::cerr << "Bad " << __FILE__ << "  " << __LINE__ << "\n";
    return *(Split<Simplex>*)nullptr;
  }






  }//eops



}//namespace dray

#endif//DRAY_ISO_OPS_HPP
