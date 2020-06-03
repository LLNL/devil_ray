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
#include <dray/Element/elem_attr.hpp>
#include <dray/Element/elem_ops.hpp>
#include <dray/Element/subref.hpp>

#include <bitset>

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

  // internal, will be undef'd at end of file.
#ifdef DRAY_CUDA_ENABLED
#define THROW_LOGIC_ERROR(msg) assert(!(msg) && false);
#else
#define THROW_LOGIC_ERROR(msg) throw std::logic_error(msg);
#endif


  namespace eops
  {

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

  DRAY_EXEC int8 isosign(Float value, Float isovalue)
  {
    return (value < isovalue - epsilon<Float>() ? -1
            : value > isovalue + epsilon<Float>() ? +1
            : 0);
  }

  // TODO use sampling because the control points will be too generous.
  template <class RotatedIndexT>
  DRAY_EXEC int32 edge_var(const RotatedIndexT &wheel, const ScalarDP &dofs, Float iota, int32 p)
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
  DRAY_EXEC bool face_cut_hex(const RotatedIndexT &wheel, const ScalarDP &dofs, Float iota, int32 p)
  {
    Range dof_range;
    for (int j = 0; j <=p; ++j)
      for (int i = 0; i <=p; ++i)
        dof_range.include(dofs[wheel.linearize(i,j)][0]);
    return dof_range.contains(iota);
  }

  DRAY_EXEC bool int_cut_hex(const ScalarDP &dofs, Float iota, int32 p)
  {
    Range dof_range;
    const int32 ndofs = (p+1)*(p+1)*(p+1);
    for (int i = 0; i < ndofs; ++i)
      dof_range.include(dofs[i][0]);
    return dof_range.contains(iota);
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

    DRAY_EXEC void clear() { m_cut_type_flag = 0;  m_bad_faces_flag = 0;  m_bad_edges_flag = 0; }
  };
  std::ostream & operator<<(std::ostream &out, const IsocutInfo &ici);

  struct CutEdges
  {
    uint32 cut_edges;
    uint32 complex_edges;
  };

  DRAY_EXEC CutEdges get_cut_edges(ShapeHex, const ScalarDP & dofs, Float iota, int32 p)
  {
    const HexFlat hlin{p};

    using namespace hex_flags;

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

    CutEdges edge_flags;
    edge_flags.cut_edges = ce;
    edge_flags.complex_edges = be;
    return edge_flags;
  }


  DRAY_EXEC uint8 get_cut_faces(ShapeHex, const ScalarDP & dofs, Float iota, int32 p)
  {
    const HexFlat hlin{p};
    using namespace hex_flags;

    uint8 cf = 0;
    cf |= f04 * face_cut_hex(RotatedIdx3<0,1,2, HexFlat>(0,0,0, hlin), dofs, iota, p);
    cf |= f05 * face_cut_hex(RotatedIdx3<0,1,2, HexFlat>(0,0,p, hlin), dofs, iota, p);
    cf |= f00 * face_cut_hex(RotatedIdx3<1,2,0, HexFlat>(0,0,0, hlin), dofs, iota, p);
    cf |= f01 * face_cut_hex(RotatedIdx3<1,2,0, HexFlat>(p,0,0, hlin), dofs, iota, p);
    cf |= f02 * face_cut_hex(RotatedIdx3<2,0,1, HexFlat>(0,0,0, hlin), dofs, iota, p);
    cf |= f03 * face_cut_hex(RotatedIdx3<2,0,1, HexFlat>(0,p,0, hlin), dofs, iota, p);

    return cf;
  }


  DRAY_EXEC IsocutInfo measure_isocut(ShapeHex, const ScalarDP & dofs, Float iota, int32 p)
  {
    IsocutInfo info;
    info.clear();

    using namespace hex_flags;

    // All cut edges and "bad" edges (bad = cut more than once).
    CutEdges edge_flags = get_cut_edges(ShapeHex(), dofs, iota, p);
    const uint32 &ce = edge_flags.cut_edges;
    const uint32 &be = edge_flags.complex_edges;

    // Update info with edges.
    info.m_bad_edges_flag = be;
    info.m_cut_type_flag |= IsocutInfo::EdgeManyPoint * bool(be);

    // All cut faces.
    const uint8 cf = get_cut_faces(ShapeHex(), dofs, iota, p);

    // FaceNoEdge (A face that is cut without any of its edges being cut).
    uint8 fne = 0;
    fne |= f04 * ((cf & f04) && !(ce & (e00 | e01 | e04 | e05)));
    fne |= f05 * ((cf & f05) && !(ce & (e02 | e03 | e06 | e07)));
    fne |= f00 * ((cf & f00) && !(ce & (e04 | e06 | e08 | e10)));
    fne |= f01 * ((cf & f01) && !(ce & (e05 | e07 | e09 | e11)));
    fne |= f02 * ((cf & f02) && !(ce & (e00 | e02 | e08 | e09)));
    fne |= f03 * ((cf & f03) && !(ce & (e01 | e03 | e10 | e11)));

    // FaceManyEdge (A face for which more than two incident edges are cut).
    uint8 fme = 0;
    fme |= f04 * (bool(ce & e00) + bool(ce & e01) + bool(ce & e04) + bool(ce & e05) > 2);
    fme |= f05 * (bool(ce & e02) + bool(ce & e03) + bool(ce & e06) + bool(ce & e07) > 2);
    fme |= f00 * (bool(ce & e04) + bool(ce & e06) + bool(ce & e08) + bool(ce & e10) > 2);
    fme |= f01 * (bool(ce & e05) + bool(ce & e07) + bool(ce & e09) + bool(ce & e11) > 2);
    fme |= f02 * (bool(ce & e00) + bool(ce & e02) + bool(ce & e08) + bool(ce & e09) > 2);
    fme |= f03 * (bool(ce & e01) + bool(ce & e03) + bool(ce & e10) + bool(ce & e11) > 2);

    // Update info with faces.
    info.m_bad_faces_flag |= fne | fme;
    info.m_cut_type_flag |= IsocutInfo::FaceNoEdge * bool(fne);
    info.m_cut_type_flag |= IsocutInfo::FaceManyEdge * bool(fme);

    const int8 num_cut_faces
      = (uint8(0) + bool(cf & f04) + bool(cf & f05) + bool(cf & f00)
                  + bool(cf & f01) + bool(cf & f02) + bool(cf & f03));

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


  DRAY_EXEC Split<Tensor> pick_iso_simple_split(ShapeHex, const IsocutInfo &info)
  {
    using namespace hex_flags;

    const uint8 &bf = info.m_bad_faces_flag;
    const uint32 &be = info.m_bad_edges_flag;

    Float score_x = 0, score_y = 0, score_z = 0;

    // Problematic edges on an axis increase likelihood that axis is split.
    score_x += 0.25f * (bool(be & e00) + bool(be & e01) + bool(be & e02) + bool(be & e03));
    score_y += 0.25f * (bool(be & e04) + bool(be & e05) + bool(be & e06) + bool(be & e07));
    score_z += 0.25f * (bool(be & e08) + bool(be & e09) + bool(be & e10) + bool(be & e11));

    // Problematic faces normal to an axis decrease likelihood that axis is split.
    score_x -= 0.5f * (bool(bf & f00) + bool(bf & f01));
    score_y -= 0.5f * (bool(bf & f02) + bool(bf & f03));
    score_z -= 0.5f * (bool(bf & f04) + bool(bf & f05));

    const int32 split_axis = (score_x > score_y ?
                               (score_x > score_z ? 0 : 2) :
                               (score_y > score_z ? 1 : 2));

    return Split<Tensor>::half(split_axis);
  }




  DRAY_EXEC IsocutInfo measure_isocut(ShapeTet, const ScalarDP & dofs, Float iota, int32 p)
  {
    THROW_LOGIC_ERROR("Not implemented in " __FILE__ " measure_isocut(ShapeTet)")
    return *(IsocutInfo*)nullptr;
  }


  DRAY_EXEC Split<Simplex> pick_iso_simple_split(ShapeTet, const IsocutInfo &info)
  {
    THROW_LOGIC_ERROR("Not implemented in " __FILE__ " pick_iso_simple_split(ShapeTet)")
    return *(Split<Simplex>*)nullptr;
  }



  // TODO These routines might not handle degeneracies gracefully. Need symbolic perturbations.

  DRAY_EXEC Float isointercept_linear(const Vec<Float, 1> &v0,
                                      const Vec<Float, 1> &v1,
                                      Float iota)
  {
    // Assume there exists t such that:
    //     v0 * (1-t) + v1 * (t) == iota
    //     <--> t * (v1-v0) == (iota-v0)
    //     <--> t = (iota-v0)/(v1-v0)  or  v0==v1
    const Float delta = v1[0] - v0[0];
    iota -= v0[0];
    return iota / delta;
  }

  DRAY_EXEC Float isointercept_quadratic(const Vec<Float, 1> &v0,
                                         const Vec<Float, 1> &v1,
                                         const Vec<Float, 1> &v2,
                                         Float iota)
  {
    // Assume there exists t such that:
    //     v0 * (1-t)^2  +  v1 * 2*(1-t)*t  +  v2 * (t)^2  == iota
    //     <--> t^2 * (v2 - 2*v1 + v0) + t * 2*(v1 - v0) == (iota-v0)
    //              dd20:=(v2 - 2*v1 + v0)      d10:=(v1-v0)
    //
    //      --> t = -(d10/dd20) +/- sqrt[(iota-v0)/dd20 + (d10/dd20)^2]
    //
    const Float d10 = v1[0] - v0[0];
    const Float dd20 = v2[0] - 2*v1[0] + v0[0];
    iota -= v0[0];
    const Float x = -d10/dd20;
    const Float w = sqrt(iota/dd20 + (x*x));
    const Float tA = x+w;
    const Float tB = x-w;
    // If one root is in the unit interval, pick it.
    return (fabs(tA-0.5) <= fabs(tB-0.5) ? tA : tB);
  }

  /// template <typename EdgeLinearizerT>
  /// DRAY_EXEC Float eval_edge_d(const ScalarDP &dofs_in,
  ///                             EdgeLinearizerT elin,
  ///                             int32 p,
  ///                             Float &J)
  /// {
  ///   // TODO implement EdgeLinearizer and then steal eval_d() from element
  /// }

  /// template <typename EdgeLinearizerT>
  /// DRAY_EXEC Float isointercept_general(const ScalarDP &dofs_in,
  ///                                      EdgeLinearizerT elin,
  ///                                      Float iota,
  ///                                      int32 p)
  /// {
  ///   Float t, f, J;

  ///   // Initial guess should be near the crossing.
  ///   int8 v_lo = 0, v_hi = p;
  ///   const bool sign_lo = elin(dofs_in, v_lo) >= iota;
  ///   const bool sign_hi = elin(dofs_in, v_hi) >= iota;
  ///   while (v_lo < p && (elin(dofs_in, v_lo+1) >= iota) == sign_lo)
  ///     v_lo++;
  ///   while (v_hi > 0 && (elin(dofs_in, v_hi-1) >= iota) == sign_hi)
  ///     v_hi--;
  ///   t = 0.5f * (v_lo + v_hi) / p;

  ///   // Do N Newton--Raphson steps.
  ///   const int8 N = 8;
  ///   for (int8 step = 0; step < N; step++)
  ///   {
  ///     f = eval_edge_d(dofs_in, elin, p, J);
  ///     t += (iota-f)/J;
  ///   }
  ///   return t;
  /// }


  DRAY_EXEC Float cut_edge_hex(const uint8 eid, const ScalarDP &C, Float iota, const OrderPolicy<1> order_p)
  {
    constexpr uint8 p = eattr::get_order(order_p.as_cxp());
    const int32 off0 = hex_props::hex_eoffset0(eid);
    const int32 off1 = hex_props::hex_eoffset1(eid);
    const int32 off2 = hex_props::hex_eoffset2(eid);
    const int32 offset = p*(p+1)*(p+1)*off2 + p*(p+1)*off1 + p*off0;
    const int32 stride = hex_props::hex_estride(eid, p+1);

    return isointercept_linear(C[offset + 0*stride], C[offset + 1*stride], iota);
  }

  DRAY_EXEC Float cut_edge_hex(const uint8 eid, const ScalarDP &C, Float iota, const OrderPolicy<2> order_p)
  {
    constexpr uint8 p = eattr::get_order(order_p.as_cxp());
    const int32 off0 = hex_props::hex_eoffset0(eid);
    const int32 off1 = hex_props::hex_eoffset1(eid);
    const int32 off2 = hex_props::hex_eoffset2(eid);
    const int32 offset = p*(p+1)*(p+1)*off2 + p*(p+1)*off1 + p*off0;
    const int32 stride = hex_props::hex_estride(eid, p+1);

    return isointercept_quadratic(C[offset + 0*stride],
                                  C[offset + 1*stride],
                                  C[offset + 2*stride], iota);
  }

  DRAY_EXEC Float cut_edge_hex(const uint8 eid, const ScalarDP &C, Float iota, const OrderPolicy<-1> order_p)
  {
    THROW_LOGIC_ERROR("Not implemented in " __FILE__ " cut_edge_hex(..., OrderPolicy<-1>")
    return -500;
  }


  /**
   * @brief Solve for the reference coordinates of a triangular isopatch inside a hex.
   */
  template <int32 P>
  DRAY_EXEC void reconstruct_isopatch(ShapeHex, ShapeTri,
      const ScalarDP & in,
      WriteDofPtr<Vec<Float, 3>> & out,
      Float iota,
      OrderPolicy<P> order_p)
  {
    // Since the isocut is 'simple,' there is a very restricted set of cases.
    // Each cut face has exactly two cut edges: Cell faces -> patch edges.
    // For tri patch, there are 3 cut faces and 3 cut edges.
    // Among the cut faces, each pair must share an edge. Opposing faces eliminated.
    // Thus from (6 choose 3)==20 face combos, 12 are eliminated, leaving 8.
    // These 8 correspond to 3sets joined by a common vertex.
    //
    // 000: X0.Y0.Z0     (f2.f4.f0)
    // 001: X1.Y0.Z0     (f3.f4.f0)
    // ...
    // 111: X1.Y1.Z1     (f3.f5.f1)

    using namespace hex_flags;

    /// constexpr uint8 caseF000 = f00|f02|f04;  constexpr uint32 caseE000 = e00|e04|e08;
    /// constexpr uint8 caseF001 = f01|f02|f04;  constexpr uint32 caseE001 = e00|e05|e09;
    /// constexpr uint8 caseF010 = f00|f03|f04;  constexpr uint32 caseE010 = e01|e04|e10;
    /// constexpr uint8 caseF011 = f01|f03|f04;  constexpr uint32 caseE011 = e01|e05|e11;
    /// constexpr uint8 caseF100 = f00|f02|f05;  constexpr uint32 caseE100 = e02|e06|e08;
    /// constexpr uint8 caseF101 = f01|f02|f05;  constexpr uint32 caseE101 = e02|e07|e09;
    /// constexpr uint8 caseF110 = f00|f03|f05;  constexpr uint32 caseE110 = e03|e06|e10;
    /// constexpr uint8 caseF111 = f01|f03|f05;  constexpr uint32 caseE111 = e03|e07|e11;

    const int32 p = eattr::get_order(order_p);

    const uint32 cut_edges = get_cut_edges(ShapeHex(), in, iota, p).cut_edges;
    const uint8 cut_faces = get_cut_faces(ShapeHex(), in, iota, p);

    using ::dray::detail::cartesian_to_tri_idx;

    // For each cell edge, solve for isovalue intercept along the edge.
    // This is univariate root finding for an isolated single root.
    // --> Vertices of the isopatch.
    uint8 edge_ids[3];
    Float edge_split[3];

    if (!(cut_edges & (e00 | e01 | e02 | e03)))
      THROW_LOGIC_ERROR("Hex->Tri: No X edges (" __FILE__ ")")
    if (!(cut_edges & (e04 | e05 | e06 | e07)))
      THROW_LOGIC_ERROR("Hex->Tri: No Y edges (" __FILE__ ")")
    if (!(cut_edges & (e08 | e09 | e10 | e11)))
      THROW_LOGIC_ERROR("Hex->Tri: No Z edges (" __FILE__ ")")

    edge_ids[0] = (cut_edges & e00) ? 0 : (cut_edges & e01) ? 1 : (cut_edges & e02) ? 2 : 3;
    edge_ids[1] = (cut_edges & e04) ? 4 : (cut_edges & e05) ? 5 : (cut_edges & e06) ? 6 : 7;
    edge_ids[2] = (cut_edges & e08) ? 8 : (cut_edges & e09) ? 9 : (cut_edges & e10) ? 10 : 11;

    edge_split[0] = cut_edge_hex(edge_ids[0], in, iota, order_p);
    edge_split[1] = cut_edge_hex(edge_ids[1], in, iota, order_p);
    edge_split[2] = cut_edge_hex(edge_ids[2], in, iota, order_p);

    // triW:edge_ids[0]  triX:edge_ids[1]  triY:edge_ids[2]
    const Vec<Float, 3> vW = {{edge_split[0], 1.0f*bool(cut_edges & (e01 | e03)), 1.0f*bool(cut_edges & (e02 | e03))}};
    const Vec<Float, 3> vX = {{1.0f*bool(cut_edges & (e05 | e07)), edge_split[1], 1.0f*bool(cut_edges & (e06 | e07))}};
    const Vec<Float, 3> vY = {{1.0f*bool(cut_edges & (e09 | e11)), 1.0f*bool(cut_edges & (e10 | e11)), edge_split[2]}};

    out[cartesian_to_tri_idx(0,0,p+1)] = vW;
    out[cartesian_to_tri_idx(p,0,p+1)] = vX;
    out[cartesian_to_tri_idx(0,p,p+1)] = vY;


    // For each cell face, solve for points in middle of isocontour within the face.
    // --> Boundary edges the isopatch.

    // Set initial guesses for patch edges (linear).
    for (uint8 i = 1; i < p; ++i)
    {
      out[cartesian_to_tri_idx(i, 0, p+1)]   = (vW*(p-i) + vX*i)/p;   // Tri edge W-->0
    }
    for (uint8 i = 1; i < p; ++i)
    {
      out[cartesian_to_tri_idx(0, i, p+1)]   = (vW*(p-i) + vY*i)/p;   // Tri edge W-->1
      out[cartesian_to_tri_idx(p-i, i, p+1)] = (vX*(p-i) + vY*i)/p;   // Tri edge 0-->1
    }


    // Solve for edge interiors.
    for (uint8 patch_edge_idx = 0; patch_edge_idx < 3; ++patch_edge_idx)
    {
      // Need patch edge id and cell face id to convert coordinates.
      constexpr tri_props::EdgeIds edge_list[3] = { tri_props::edgeW0,
                                                    tri_props::edgeW1,
                                                    tri_props::edge01 };
      const uint8 patch_edge = edge_list[patch_edge_idx];

      constexpr uint8 te_end0[3] = {0, 0, 1};
      constexpr uint8 te_end1[3] = {1, 2, 2};

      const uint8 fid = hex_props::hex_common_face(
          edge_ids[te_end0[patch_edge_idx]], edge_ids[te_end1[patch_edge_idx]] );
      const HexFaceWalker<P> cell_fw(order_p, fid);
      const TriEdgeWalker<P> patch_ew(order_p, patch_edge);

      // Solve for each dof in the edge.
      for (int32 i = 1; i < p; ++i)
      {
        // Get initial guess.
        Vec<Float, 3> pt3 = out[patch_ew.edge2tri(i)];
        Vec<Float, 2> pt2 = {{pt3[hex_props::hex_faxisU(fid)],
                              pt3[hex_props::hex_faxisV(fid)]}};

        const Vec<Float, 2> pt2_next =
            {{ out[patch_ew.edge2tri(i+1)][hex_props::hex_faxisU(fid)],
               out[patch_ew.edge2tri(i+1)][hex_props::hex_faxisV(fid)] }};

          // Variant "13" is from the paper: init by normal on linear.
        const Vec<Float, 2> init_dir_v13 =
            (Vec<Float, 2>{{-pt2_next[1], pt2_next[0]}}).normalized();

        Vec<Float, 2> search_dir = init_dir_v13;

        // Do a few iterations.
        constexpr int32 num_iter = 5;
        for (int32 t = 0; t < num_iter; ++t)
        {
          Vec<Vec<Float, 1>, 2> deriv;
          Vec<Float, 1> scalar = eval_d_face(ShapeHex(), order_p, fid, in, pt2, deriv);
          pt2 += search_dir * ((iota-scalar[0]) * rcp_safe(dot(deriv, search_dir)[0]));
        }

        // Store the updated point.
        pt3[hex_props::hex_faxisU(fid)] = pt2[0];
        pt3[hex_props::hex_faxisV(fid)] = pt2[1];
        out[patch_ew.edge2tri(i)] = pt3;
      }
    }


    // For the cell volume, solve for points in middle of isopatch.

    //TODO init guess for patch interior

    //TODO solve patch interior

    /// switch (cut_faces)
    /// {
    ///   case caseF000: assert(cut_edges == caseE000);
    ///     break;
    ///   case caseF001: assert(cut_edges == caseE001);
    ///     break;
    ///   case caseF010: assert(cut_edges == caseE010);
    ///     break;
    ///   case caseF011: assert(cut_edges == caseE011);
    ///     break;
    ///   case caseF100: assert(cut_edges == caseE100);
    ///     break;
    ///   case caseF101: assert(cut_edges == caseE101);
    ///     break;
    ///   case caseF110: assert(cut_edges == caseE110);
    ///     break;
    ///   case caseF111: assert(cut_edges == caseE111);
    ///     break;
    ///   default:
    ///     THROW_LOGIC_ERROR("Unexpected tri isopatch case (" __FILE__ ")")
    /// }

    /// // STUB
    /// lagrange_pts_out[0] = {{0.5, 0.5, 0}};
    /// lagrange_pts_out[1] = {{0.5, 0.25, 0}};
    /// lagrange_pts_out[2] = {{0.5, 0, 0}};
    /// lagrange_pts_out[3] = {{0.25, 0.5, 0}};
    /// lagrange_pts_out[4] = {{0.25, 0.25, 0}};
    /// lagrange_pts_out[5] = {{0, 0.5, 0}};

    // TODO consider breaking out the solves for each point for higher parallelism.

  }


  /**
   * @brief Solve for the reference coordinates of a quad isopatch inside a hex.
   */
  template <int32 P>
  DRAY_EXEC void reconstruct_isopatch(ShapeHex, ShapeQuad,
      const ScalarDP & in,
      WriteDofPtr<Vec<Float, 3>> & out,
      Float iota,
      OrderPolicy<P> order_p)
  {
    // Since the isocut is 'simple,' there is a very restricted set of cases.
    // Each cut face has exactly two cut edges: Cell faces -> patch edges.
    // For quad patch, there are 4 cut faces and 4 cut edges.
    // All (6 choose 4)==15 combos are valid cuts.
    // There are two types: Axis-aligned into 2 cubes --> x3 axes
    //                      Corner-cutting into prism --> x4 corners x3 axes.

    const int32 p = eattr::get_order(order_p);

    const uint32 cut_edges = get_cut_edges(ShapeHex(), in, iota, p).cut_edges;
    const uint8 cut_faces = get_cut_faces(ShapeHex(), in, iota, p);

    using namespace hex_flags;

    uint8 edge_ids[4];
    uint8 split_counter = 0;
    for (uint8 e = 0; e < 12; ++e)
      if ((cut_edges & (1u<<e)))
        edge_ids[split_counter++] = e;

    // Corners of the patch live on cell edges.
    Vec<Float, 3> corners[4];
    for (uint8 s = 0; s < 4; ++s)
    {
      const uint8 e = edge_ids[s];

      // Get base coordinate of cell edge.
      corners[s][0] = hex_props::hex_eoffset0(e);
      corners[s][1] = hex_props::hex_eoffset1(e);
      corners[s][2] = hex_props::hex_eoffset2(e);

      // Overwrite coordinate along the cell edge based on iso cut.
      corners[s][hex_props::hex_eaxis(e)] = cut_edge_hex(e, in, iota, order_p);
    }

    out[(p+1)*0 + 0] = corners[0];
    out[(p+1)*0 + p] = corners[1];
    out[(p+1)*p + 0] = corners[2];
    out[(p+1)*p + p] = corners[3];

    // Set initial guesses for patch edges (linear).
    for (uint8 i = 1; i < p; ++i)
    {
      out[(p+1)*0 + i] = (corners[0]*(p-1) + corners[1]*i)/p;  // Quad edge 0
    }
    for (uint8 i = 1; i < p; ++i)
    {
      out[(p+1)*i + 0] = (corners[0]*(p-i) + corners[2]*i)/p;  // Quad edge 2
      out[(p+1)*i + p] = (corners[1]*(p-i) + corners[3]*i)/p;  // Quad edge 3
    }
    for (uint8 i = 1; i < p; ++i)
    {
      out[(p+1)*p + i] = (corners[2]*(p-1) + corners[3]*i)/p;  // Quad edge 1
    }


    // Solve for edge interiors.
    for (uint8 patch_edge = 0; patch_edge < 4; ++patch_edge)
    {
      constexpr uint8 qe_end0[4] = {0, 2, 0, 1};
      constexpr uint8 qe_end1[4] = {1, 3, 2, 3};

      const uint8 fid = hex_props::hex_common_face(
          edge_ids[qe_end0[patch_edge]], edge_ids[qe_end1[patch_edge]] );
      const HexFaceWalker<P> cell_fw(order_p, fid);
      const QuadEdgeWalker<P> patch_ew(order_p, patch_edge);

      // For now, move each point individually.
      // TODO coordination to get optimal spacing.
      for (int32 i = 1; i < p; ++i)
      {
        // Get initial guess.
        Vec<Float, 3> pt3 = out[patch_ew.edge2quad(i)];
        Vec<Float, 2> pt2 = {{pt3[hex_props::hex_faxisU(fid)],
                              pt3[hex_props::hex_faxisV(fid)]}};

        const Vec<Float, 2> pt2_next =
            {{ out[patch_ew.edge2quad(i+1)][hex_props::hex_faxisU(fid)],
               out[patch_ew.edge2quad(i+1)][hex_props::hex_faxisV(fid)] }};

        // Set up search direction. There are several options here.
        // One is to use a fixed search direction based on the initial guess.
        // Another option is to do gradient descent.
        const Vec<Float, 2> init_dir_v13 =
            (Vec<Float, 2>{{-pt2_next[1], pt2_next[0]}}).normalized();

        Vec<Float, 2> search_dir = init_dir_v13;

        // Do a few iterations.
        constexpr int32 num_iter = 5;
        for (int32 t = 0; t < num_iter; ++t)
        {
          Vec<Vec<Float, 1>, 2> deriv;
          Vec<Float, 1> scalar = eval_d_face(ShapeHex(), order_p, fid, in, pt2, deriv);
          pt2 += search_dir * ((iota-scalar[0]) * rcp_safe(dot(deriv, search_dir)[0]));
        }

        // Store the updated point.
        pt3[hex_props::hex_faxisU(fid)] = pt2[0];
        pt3[hex_props::hex_faxisV(fid)] = pt2[1];
        out[patch_ew.edge2quad(i)] = pt3;
      }
    }

    // Initial guess for patch interior.
    // Follows paper appendix formula for quads.

    // The paper (transformed to the unit square [0,1]^2) does this:
    //
    //  out(i,j) =  corners[0]*(1-xi)*(1-yj) + corners[1]*( xi )*(1-yj)   // Lerp corners
    //            + corners[2]*(1-xi)*( yj ) + corners[3]*( xi )*( yj )
    //
    //            + (out(i,0) - corners[0]*(1-xi) - corners[1]*( xi ))*(1-yj) // Eval deviation on x edges
    //            + (out(i,p) - corners[2]*(1-xi) - corners[3]*( xi ))*( yj ) //  and lerp the deviation
    //
    //            + (out(0,j) - corners[0]*(1-yj) - corners[2]*( yj ))*(1-xi) // Eval deviation on y edges
    //            + (out(p,j) - corners[1]*(1-yj) - corners[3]*( yj ))*( xi ) //  and lerp the deviation
    //
    // Many of the terms cancel, making this equivalent:
    //
    for (int32 j = 1; j < p; ++j)
    {
      const Vec<Float, 3> dof_e2 = out[(p+1)*j + 0];
      const Vec<Float, 3> dof_e3 = out[(p+1)*j + p];
      const Float yj = Float(j)/Float(p);
      const Float _yj = 1.0f - yj;

      for (int32 i = 1; i < p; ++i)
      {
        const Vec<Float, 3> dof_e0 = out[(p+1)*0 + i];
        const Vec<Float, 3> dof_e1 = out[(p+1)*p + i];
        const Float xi = Float(i)/Float(p);
        const Float _xi = 1.0f - xi;

        out[(p+1)*j + i] =  dof_e0 * _yj + dof_e1 * yj
                          + dof_e2 * _xi + dof_e3 * xi
                          - corners[0] * (_xi * _yj)
                          - corners[1] * ( xi * _yj)
                          - corners[2] * (_xi *  yj)
                          - corners[3] * ( xi *  yj);
      }
    }

    // Solve for patch interior.
    // TODO coordination for optimal spacing.
    // For now, move each point individually.
    for (int32 j = 1; j < p; ++j)
      for (int32 i = 1; i < p; ++i)
      {
        // Get initial guess.
        Vec<Float, 3> pt3 = out[(p+1)*j + i];

        Vec<Vec<Float, 1>, 3> deriv;
        Vec<Float, 1> scalar = eval_d(ShapeHex(), order_p, in, pt3, deriv);

        // For the search direction, use initial gradient direction.
        const Vec<Float, 3> search_dir =
            (Vec<Float, 3>{{deriv[0][0], deriv[1][0], deriv[2][0]}}).normalized();

        // Do a few iterations.
        constexpr int32 num_iter = 5;
        for (int32 t = 0; t < num_iter; ++t)
        {
          scalar = eval_d(ShapeHex(), order_p, in, pt3, deriv);
          pt3 += search_dir * ((iota-scalar[0]) * rcp_safe(dot(deriv, search_dir)[0]));
        }

        // Store the updated point.
        out[(p+1)*j + i] = pt3;
      }


    // TODO consider breaking out the solves for each point for higher parallelism.

    // For each cell edge, solve for isovalue intercept along the edge.
    // This is univariate root finding for an isolated single root.
    // --> Vertices of the isopatch.

    // For each cell face, solve for points in middle of isocontour within the face.
    // --> Boundary edges the isopatch.

    // For the cell volume, solve for points in middle of isopatch.
  }


  /**
   * @brief Solve for the reference coordinates of a triangular isopatch inside a tet.
   */
  template <int32 P>
  DRAY_EXEC void reconstruct_isopatch(ShapeTet, ShapeTri,
      const ScalarDP & dofs_in,
      WriteDofPtr<Vec<Float, 3>> & lagrange_pts_out,
      Float iota,
      OrderPolicy<P> order_p)
  {
    THROW_LOGIC_ERROR("Not implemented in " __FILE__ " reconstruct_isopatch(ShapeTet, ShapeTri)")
  }

  /**
   * @brief Solve for the reference coordinates of a quad isopatch inside a tet.
   */
  template <int32 P>
  DRAY_EXEC void reconstruct_isopatch(ShapeTet, ShapeQuad,
      const ScalarDP & dofs_in,
      WriteDofPtr<Vec<Float, 3>> & lagrange_pts_out,
      Float iota,
      OrderPolicy<P> order_p)
  {
    THROW_LOGIC_ERROR("Not implemented in " __FILE__ " reconstruct_isopatch(ShapeTet, ShapeQuad)")
  }



  }//eops


#undef THROW_LOGIC_ERROR

}//namespace dray

#endif//DRAY_ISO_OPS_HPP
