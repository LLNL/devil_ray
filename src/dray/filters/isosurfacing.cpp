// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <dray/filters/isosurfacing.hpp>
#include <dray/error.hpp>
#include <dray/dispatcher.hpp>
#include <dray/array_utils.hpp>

#include <RAJA/RAJA.hpp>
#include <dray/policies.hpp>
#include <dray/exports.hpp>

#include <dray/derived_topology.hpp>
#include <dray/GridFunction/mesh.hpp>
#include <dray/GridFunction/field.hpp>
#include <dray/GridFunction/device_mesh.hpp>
#include <dray/GridFunction/device_field.hpp>
#include <dray/Element/elem_attr.hpp>
#include <dray/Element/iso_ops.hpp>
#include <dray/Element/detached_element.hpp>

#include <sstream>

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

namespace dray
{
  // -----------------------
  // Getter/setters
  // -----------------------
  void ExtractIsosurface::iso_field(const std::string field_name)
  {
    m_iso_field_name = field_name;
  }

  std::string ExtractIsosurface::iso_field() const
  {
    return m_iso_field_name;
  }

  void ExtractIsosurface::iso_value(const float32 iso_value)
  {
    m_iso_value = iso_value;
  }

  Float ExtractIsosurface::iso_value() const
  {
    return m_iso_value;
  }
  // -----------------------

  namespace detail
  {
    template <typename T>
    DRAY_EXEC static void stack_push(T stack[], int32 &stack_sz, const T &item) 
    {
      stack[stack_sz++] = item;
    }

    template <typename T>
    DRAY_EXEC static T stack_pop(T stack[], int32 &stack_sz)
    {
      return stack[--stack_sz];
    }

    template <typename T>
    DRAY_EXEC static void circ_enqueue(T queue[], const int32 arrsz, int32 &q_begin, int32 &q_sz, const T &item)
    {
      queue[(q_begin + (q_sz++)) % arrsz] = item;
    }

    template <typename T>
    DRAY_EXEC static T circ_dequeue(T queue[], const int32 arrsz, int32 &q_begin, int32 &q_sz)
    {
      int32 idx = q_begin;
      q_sz--;
      q_begin = (q_begin + 1 == arrsz ? 0 : q_begin + 1);
      return queue[idx];
    }

    template <typename T>
    DRAY_EXEC static void circ_queue_rotate(T queue[], const int32 arrsz, int32 &q_begin, int32 &q_sz)
    {
      T tmp = circ_dequeue(queue, arrsz, q_begin, q_sz);
      circ_enqueue(queue, arrsz, q_begin, q_sz, tmp);
    }
  }



  // LocationSet for remapping non-iso fields.
  struct LocationSet
  {
    GridFunction<3> m_rcoords;     // one per dof per element.
    Array<int32> m_host_cell_id;   // one per element.
  };

  template <typename OutShape, int32 MP, class FElemT>
  std::shared_ptr<FieldBase> ReMapField_execute(const LocationSet &location_set,
                                                OutShape out_shape,
                                                OrderPolicy<MP> mesh_order_p,
                                                Field<FElemT> &in_field);

  // ReMapFieldFunctor
  template <typename OutShape, int32 P>
  struct ReMapFieldFunctor
  {
    LocationSet m_location_set;
    OrderPolicy<P> m_mesh_order_p;

    std::shared_ptr<FieldBase> m_out_field_ptr;

    ReMapFieldFunctor(const LocationSet &ls, OrderPolicy<P> mesh_order_p)
      : m_location_set(ls),
        m_mesh_order_p(mesh_order_p),
        m_out_field_ptr(nullptr)
    { }

    template <typename FieldT>
    void operator()(FieldT &field)
    {
      m_out_field_ptr = ReMapField_execute(m_location_set, OutShape(), m_mesh_order_p, field);
    }
  };




  //
  // execute(topo, field)
  //
  template <class MElemT, class FElemT>
  std::pair<DataSet,DataSet> ExtractIsosurface_execute( DerivedTopology<MElemT> &topo,
                                                        Field<FElemT> &field,
                                                        Float iso_value,
                                                        DataSet *input_dataset)
  {
    // Overview:
    //   - Subdivide field elements until isocut is simple. Yields sub-ref and sub-coeffs.
    //   - Extract isopatch coordinates relative to the coord-system of sub-ref.
    //   - Transform isopatch coordinates to reference space via (sub-ref)^{-1}.
    //   - Transform isopatch coordinates to world space via mesh element.
    //   - Outside this function, the coordinates are converted to Bernstein ctrl points.

    static_assert(FElemT::get_ncomp() == 1, "Can't take isosurface of a vector field");

    const Float isoval = iso_value;  // Local for capture
    const int32 n_el_in = field.get_num_elem();
    DeviceField<FElemT> dfield(field);

    constexpr int32 budget = 10;
    Array<uint8> count_subelem_required;
    Array<uint8> budget_maxed;

    count_subelem_required.resize(n_el_in);
    budget_maxed.resize(n_el_in);
    uint8 *count_required_ptr = count_subelem_required.get_device_ptr();
    uint8 *budget_maxed_ptr = budget_maxed.get_device_ptr();

    const auto field_order_p = dfield.get_order_policy();
    const auto shape3d = adapt_get_shape(FElemT{});
    const int32 field3d_npe = eattr::get_num_dofs(shape3d, field_order_p);

    GridFunction<1> field_sub_elems;
    field_sub_elems.resize_counting(n_el_in * budget, field3d_npe);
    const int32 * f_subel_iptr = field_sub_elems.m_ctrl_idx.get_device_ptr();
    Vec<Float, 1> * f_subel_vptr = field_sub_elems.m_values.get_device_ptr();

    using SubRefT = typename get_subref<FElemT>::type;
    Array<SubRefT> subref_array;
    subref_array.resize(n_el_in * budget);
    SubRefT * subref_ptr = subref_array.get_device_ptr();

    Array<int32> keepme_tri, keepme_quad;
    keepme_tri.resize(n_el_in * budget);
    keepme_quad.resize(n_el_in * budget);
    array_memset_zero(keepme_tri);
    array_memset_zero(keepme_quad);
    int32 *keepme_tri_ptr = keepme_tri.get_device_ptr();
    int32 *keepme_quad_ptr = keepme_quad.get_device_ptr();

    RAJA::forall<for_policy>(RAJA::RangeSegment(0, n_el_in), [=] DRAY_LAMBDA (int32 eidx) {
        using eops::measure_isocut;
        using eops::IsocutInfo;

        using WDP = WriteDofPtr<Vec<Float, 1>>;

        FElemT felem_in = dfield.get_elem(eidx);

        const auto shape = adapt_get_shape(felem_in);
        constexpr ElemType etype = eattr::get_etype(shape);
        const RefSpaceTag<3, etype> ref_space_tag;
        const auto forder_p = dfield.get_order_policy();
        const int32 fp = eattr::get_order(forder_p);

        // Breadth-first subdivision (no priorities)
        SubRef<3, etype> * subrefs = subref_ptr + eidx * budget;
        WDP felem_store[budget];
        for (int32 f = 0; f < budget; ++f)
        {
          felem_store[f].m_offset_ptr = f_subel_iptr + (eidx * budget + f) * field3d_npe;
          felem_store[f].m_dof_ptr = f_subel_vptr;
        }
        uint32 occupied = 0u;
        IsocutInfo info_store[budget];

        int8 pool[budget];
        for (int32 f = 0; f < budget; ++f)
          pool[f] = budget-1-f;   // {9, 8, 7, 6, 5, 4, 3, 2, 1, 0};
        int32 pool_sz = budget;
        int8 q[budget];
        int32 q_begin = 0;
        int32 q_sz = 0;
        bool stuck = false;
        int8 stuck_counter = 0;
        int8 ceiling = 0;
        int8 qceil = 0;

        using detail::stack_push;
        using detail::stack_pop;
        using detail::circ_enqueue;
        using detail::circ_dequeue;
        using detail::circ_queue_rotate;

        // Enqueue the input element.
        circ_enqueue(q, budget, q_begin, q_sz,  stack_pop(pool, pool_sz));
        occupied |= (1u << q[q_begin]);

        subrefs[ q[q_begin] ] = ref_universe(ref_space_tag);

        for (int32 nidx = 0; nidx < field3d_npe; ++nidx)
          felem_store[ q[q_begin] ][nidx] = felem_in.read_dof_ptr()[nidx];

        info_store[ q[q_begin] ] = measure_isocut(shape,
            felem_store[ q[q_begin] ].to_readonly_dof_ptr(), isoval, fp);

        // Test each element in the queue.
        //   If no cut, discard and return slot from store to pool.
        //   If simple cut, keep slot reserved in store.
        //   If non simple cut, perform a split if there is room to do so.
        while ( q_sz > 0 )
        {
          ceiling = (budget-pool_sz > ceiling ? budget-pool_sz : ceiling);
          qceil = (q_sz > qceil ? q_sz : qceil);

          const IsocutInfo &head_info = info_store[ q[q_begin] ];

          if (head_info.m_cut_type_flag == 0)
          {
            /// dbgout << "[" << eidx << "] No cut.\n";
            int8 dropped_qitem = circ_dequeue(q, budget, q_begin, q_sz);
            occupied &= ~(1u << dropped_qitem);
            stack_push(pool, pool_sz, dropped_qitem);  // restore to circulation.

            stuck = false;
            stuck_counter = 0;
          }
          else if (head_info.m_cut_type_flag < 8)
          {
            /// if (head_info.m_cut_type_flag & IsocutInfo::CutSimpleTri)
            ///   dbgout << "[" << eidx << "] Simple tri\n";
            /// else if (head_info.m_cut_type_flag & IsocutInfo::CutSimpleQuad)
            ///   dbgout << "[" << eidx << "] Simple quad\n";

            circ_dequeue(q, budget, q_begin, q_sz);  // Reserve and remove from circulation.
            // 'occupied' remains same, still occupies slot in store.

            stuck = false;
            stuck_counter = 0;
          }
          else
          {
            if (pool_sz > 0)
            {
              // Prepare to split
              const int8 mother = circ_dequeue(q, budget, q_begin, q_sz);
              const int8 daughter = stack_pop(pool, pool_sz);
              occupied |= (1u << daughter);
              // occupied is already set for mother.

              for (int32 nidx = 0; nidx < field3d_npe; ++nidx)
                felem_store[daughter][nidx] = felem_store[mother][nidx];

              // Get the split direction.
              Split<etype> binary_split = pick_iso_simple_split(shape, info_store[mother]);
              /// dbgout << "[" << eidx << "] Splitting... " << binary_split << "\n";

              // Perform the split.
              subrefs[daughter] = split_subref(subrefs[mother], binary_split);

              split_inplace(shape, forder_p, felem_store[mother], binary_split);
              split_inplace(shape, forder_p, felem_store[daughter], binary_split.get_complement());

              // Update isocut info.
              info_store[mother] = measure_isocut(shape,
                  felem_store[mother].to_readonly_dof_ptr(), isoval, fp);
              info_store[daughter] = measure_isocut(shape,
                  felem_store[daughter].to_readonly_dof_ptr(), isoval, fp);

              // Enqueue for further processing.
              circ_enqueue(q, budget, q_begin, q_sz, mother);
              circ_enqueue(q, budget, q_begin, q_sz, daughter);
            }
            else
            {
              stuck = true;
              if (stuck_counter < q_sz)  // There may still be hope
              {
                circ_queue_rotate(q, budget, q_begin, q_sz);
                stuck_counter++;
                /// dbgout << "  ..stuck (" << int32(stuck_counter) << ")\n";
              }
              else  // Give up
              {
                /// dbgout << "Giving up!\n";
                break;
              }
            }
          }
        }

        // Mark hits to be kept.
        for (int8 store_idx = 0; store_idx < budget; ++store_idx)
          if (occupied & (1u << store_idx))
          {
            if (info_store[store_idx].m_cut_type_flag & IsocutInfo::CutSimpleTri)
              keepme_tri_ptr[eidx * budget + store_idx] = true;
            else if (info_store[store_idx].m_cut_type_flag & IsocutInfo::CutSimpleQuad)
              keepme_quad_ptr[eidx * budget + store_idx] = true;
          }

        count_required_ptr[eidx] = int32(budget-pool_sz);
        budget_maxed_ptr[eidx] = bool(stuck);

        /// std::cout << "[" << eidx << "]*Finished subdividing, \t"
        ///           << "circulation (" << int32(qceil) << "/" << int32(ceiling) << "/" << int32(budget) << "),    \t"
        ///           << "final ("
        ///           << int32(budget-pool_sz) << "/" << int32(budget) << ")"
        ///           << (stuck ? "(+)" : "")
        ///           << "*\n";

    });

    GridFunction<1> field_sub_elems_tri;
    Array<SubRefT> subrefs_tri;
    Array<int32> kept_indices_tri = index_flags(keepme_tri, array_counting(keepme_tri.size(), 0, 1));
    subrefs_tri = gather(subref_array, kept_indices_tri);
    field_sub_elems_tri.m_values = gather(field_sub_elems.m_values, field3d_npe, kept_indices_tri);
    field_sub_elems_tri.m_ctrl_idx = array_counting(field3d_npe * kept_indices_tri.size(), 0, 1);
    field_sub_elems_tri.m_el_dofs = field3d_npe;
    field_sub_elems_tri.m_size_el = kept_indices_tri.size();
    field_sub_elems_tri.m_size_ctrl = field_sub_elems_tri.m_values.size();

    GridFunction<1> field_sub_elems_quad;
    Array<SubRefT> subrefs_quad;
    Array<int32> kept_indices_quad = index_flags(keepme_quad, array_counting(keepme_quad.size(), 0, 1));
    subrefs_quad = gather(subref_array, kept_indices_quad);
    field_sub_elems_quad.m_values = gather(field_sub_elems.m_values, field3d_npe, kept_indices_quad);
    field_sub_elems_quad.m_ctrl_idx = array_counting(field3d_npe * kept_indices_quad.size(), 0, 1);
    field_sub_elems_quad.m_el_dofs = field3d_npe;
    field_sub_elems_quad.m_size_el = kept_indices_quad.size();
    field_sub_elems_quad.m_size_ctrl = field_sub_elems_quad.m_values.size();

    const int32 num_sub_elems_tri = field_sub_elems_tri.m_size_el;
    const int32 num_sub_elems_quad = field_sub_elems_quad.m_size_el;

    // Now have the field values of each sub-element.
    // Create an output isopatch for each sub-element.
    // Use the FIELD order for the approximate isopatches.

    constexpr int32 out_order_policy_id = FElemT::get_P();
    const auto out_order_p = field_order_p;
    /// constexpr Order out_order_policy_id = General;     //
    /// const auto out_order_p = OrderPolicy<General>{3};  // Can use another order.
    const int32 out_order = eattr::get_order(out_order_p);
    const int32 out_tri_npe = eattr::get_num_dofs(ShapeTri(), out_order_p);
    const int32 out_quad_npe = eattr::get_num_dofs(ShapeQuad(), out_order_p);

    // Outputs for physical mesh coords of new surface elements.
    GridFunction<3> isopatch_coords_tri;
    GridFunction<3> isopatch_coords_quad;
    isopatch_coords_tri.resize_counting(num_sub_elems_tri, out_tri_npe);
    isopatch_coords_quad.resize_counting(num_sub_elems_quad, out_quad_npe);

    // Intermediate arrays to later map additional fields onto new surface elements.
    LocationSet locset_tri;
    LocationSet locset_quad;
    locset_tri.m_rcoords.resize_counting(num_sub_elems_tri, out_tri_npe);
    locset_quad.m_rcoords.resize_counting(num_sub_elems_quad, out_quad_npe);
    locset_tri.m_host_cell_id.resize(num_sub_elems_tri);
    locset_quad.m_host_cell_id.resize(num_sub_elems_quad);

    DeviceMesh<MElemT> dmesh(topo.mesh());
    const Float iota = iso_value;

    // Extract triangle isopatches.
    {
      const SubRefT * subref_ptr = subrefs_tri.get_device_ptr_const();
      DeviceGridFunction<1> field_subel_dgf(field_sub_elems_tri);
      DeviceGridFunction<3> isopatch_dgf(isopatch_coords_tri);
      const int32 *budget_idxs_tri_ptr = kept_indices_tri.get_device_ptr_const();

      int32 *host_cell_id_tri_ptr = locset_tri.m_host_cell_id.get_device_ptr();
      DeviceGridFunction<3> isopatch_r_dgf(locset_tri.m_rcoords);

      RAJA::forall<for_policy>(RAJA::RangeSegment(0, num_sub_elems_tri), [=] DRAY_LAMBDA (int32 neid) {

        ReadDofPtr<Vec<Float, 1>> field_vals = field_subel_dgf.get_rdp(neid);
        WriteDofPtr<Vec<Float, 3>> coords = isopatch_dgf.get_wdp(neid);
        eops::reconstruct_isopatch(shape3d, ShapeTri(), field_vals, coords, iota, field_order_p, out_order_p);

        const int32 host_cell_id = budget_idxs_tri_ptr[neid] / budget;
        host_cell_id_tri_ptr[neid] = host_cell_id;
        WriteDofPtr<Vec<Float, 3>> rcoords = isopatch_r_dgf.get_wdp(neid);

        const MElemT melem = dmesh.get_elem(host_cell_id);
        for (int32 nidx = 0; nidx < out_tri_npe; ++nidx)
        {
          const Vec<Float, 3> rcoord = subref2ref(subref_ptr[neid], coords[nidx]);
          rcoords[nidx] = rcoord;
          coords[nidx] = melem.eval(rcoord);
        }
      });
    }

    // Extract quad isopatches.
    {
      const SubRefT * subref_ptr = subrefs_quad.get_device_ptr_const();
      DeviceGridFunction<1> field_subel_dgf(field_sub_elems_quad);
      DeviceGridFunction<3> isopatch_dgf(isopatch_coords_quad);
      const int32 *budget_idxs_quad_ptr = kept_indices_quad.get_device_ptr_const();

      int32 *host_cell_id_quad_ptr = locset_quad.m_host_cell_id.get_device_ptr();
      DeviceGridFunction<3> isopatch_r_dgf(locset_quad.m_rcoords);

      RAJA::forall<for_policy>(RAJA::RangeSegment(0, num_sub_elems_quad), [=] DRAY_LAMBDA (int32 neid) {

        ReadDofPtr<Vec<Float, 1>> field_vals = field_subel_dgf.get_rdp(neid);
        WriteDofPtr<Vec<Float, 3>> coords = isopatch_dgf.get_wdp(neid);
        eops::reconstruct_isopatch(shape3d, ShapeQuad(), field_vals, coords, iota, field_order_p, out_order_p);

        const int32 host_cell_id = budget_idxs_quad_ptr[neid] / budget;
        host_cell_id_quad_ptr[neid] = host_cell_id;
        WriteDofPtr<Vec<Float, 3>> rcoords = isopatch_r_dgf.get_wdp(neid);

        const MElemT melem = dmesh.get_elem( budget_idxs_quad_ptr[neid] / budget );
        for (int32 nidx = 0; nidx < out_quad_npe; ++nidx)
        {
          const Vec<Float, 3> rcoord = subref2ref(subref_ptr[neid], coords[nidx]);
          rcoords[nidx] = rcoord;
          coords[nidx] = melem.eval(rcoord);
        }
      });
    }

    using IsoPatchTriT = Element<2, 3, Simplex, out_order_policy_id>;
    using IsoPatchQuadT = Element<2, 3, Tensor, out_order_policy_id>;
    Mesh<IsoPatchTriT> isosurface_tris(isopatch_coords_tri, out_order);
    Mesh<IsoPatchQuadT> isosurface_quads(isopatch_coords_quad, out_order);
    DataSet isosurface_tri_ds(std::make_shared<DerivedTopology<IsoPatchTriT>>(isosurface_tris));
    DataSet isosurface_quad_ds(std::make_shared<DerivedTopology<IsoPatchQuadT>>(isosurface_quads));

    // Remap input fields onto surfaces.
    // Need to dispatch order policy for each input field.
    ReMapFieldFunctor<ShapeTri,  out_order_policy_id> rmff_tri(locset_tri, out_order_p);
    ReMapFieldFunctor<ShapeQuad, out_order_policy_id> rmff_quad(locset_quad, out_order_p);
    for (const std::string &fname : input_dataset->fields())
    {
      dispatch_3d(input_dataset->field(fname), rmff_tri);
      dispatch_3d(input_dataset->field(fname), rmff_quad);

      isosurface_tri_ds.add_field(rmff_tri.m_out_field_ptr);
      isosurface_quad_ds.add_field(rmff_quad.m_out_field_ptr);
    }

    /// std::cout << dbgout.str() << std::endl;

    return {isosurface_tri_ds, isosurface_quad_ds};
  }


  /** remap_element() (Hex, Quad) */
  template <int32 IP, int32 MP, int32 OP, int32 ncomp>
  DRAY_EXEC void remap_element(const ShapeHex,
                               OrderPolicy<IP> in_order_p,
                               const ReadDofPtr<Vec<Float, ncomp>> &in_field_rdp,
                               const ShapeQuad,
                               OrderPolicy<MP> mesh_order_p,
                               const ReadDofPtr<Vec<Float, 3>> &mesh_rdp,
                               OrderPolicy<OP> out_order_p,
                               WriteDofPtr<Vec<Float, ncomp>> out_field_wdp)
  {
    const int32 ip = eattr::get_order(in_order_p);
    const int32 mp = eattr::get_order(mesh_order_p);
    const int32 op = eattr::get_order(out_order_p);

    //TODO use eval and don't assert equal.
    assert(mp == op);

    for (int32 j = 0; j <= op; ++j)
      for (int32 i = 0; i <= op; ++i)
      {
        Vec<Vec<Float, ncomp>, 3> UN_d = {{ {{0}}, {{0}}, {{0}} }};  // unused derivative.
        const Vec<Float, 3> host_ref_pt = mesh_rdp[j*(op+1) + i];  //TODO eval()
        const Vec<Float, ncomp> field_val =
            eops::eval_d(ShapeHex(), in_order_p, in_field_rdp, host_ref_pt, UN_d);

        out_field_wdp[j*(op+1) + i] = field_val;
      }
  }

  /** remap_element() (Hex, Tri) */
  template <int32 IP, int32 MP, int32 OP, int32 ncomp>
  DRAY_EXEC void remap_element(const ShapeHex,
                               OrderPolicy<IP> in_order_p,
                               const ReadDofPtr<Vec<Float, ncomp>> &in_field_rdp,
                               const ShapeTri,
                               OrderPolicy<MP> mesh_order_p,
                               const ReadDofPtr<Vec<Float, 3>> &mesh_rdp,
                               OrderPolicy<OP> out_order_p,
                               WriteDofPtr<Vec<Float, ncomp>> out_field_wdp)
  {
    const int32 ip = eattr::get_order(in_order_p);
    const int32 mp = eattr::get_order(mesh_order_p);
    const int32 op = eattr::get_order(out_order_p);

    //TODO use eval and don't assert equal.
    assert(mp == op);

    for (int32 j = 0; j <= op; ++j)
      for (int32 i = 0; i <= op-j; ++i)
      {
        const int32 nidx = detail::cartesian_to_tri_idx(i, j, op+1);

        Vec<Vec<Float, ncomp>, 3> UN_d = {{ {{0}}, {{0}}, {{0}} }};  // unused derivative.
        const Vec<Float, 3> host_ref_pt = mesh_rdp[nidx];  //TODO eval()
        const Vec<Float, ncomp> field_val =
            eops::eval_d(ShapeHex(), in_order_p, in_field_rdp, host_ref_pt, UN_d);

        out_field_wdp[nidx] = field_val;
      }
  }

  /** remap_element() (Tet, Quad) */
  template <int32 IP, int32 MP, int32 OP, int32 ncomp>
  DRAY_EXEC void remap_element(const ShapeTet,
                               OrderPolicy<IP> in_order_p,
                               const ReadDofPtr<Vec<Float, ncomp>> &in_field_rdp,
                               const ShapeQuad,
                               OrderPolicy<MP> mesh_order_p,
                               const ReadDofPtr<Vec<Float, 3>> &mesh_rdp,
                               OrderPolicy<OP> out_order_p,
                               WriteDofPtr<Vec<Float, ncomp>> out_field_wdp)
  {
        // TODO this is where we should evaluate instead of merely lookup.
    throw std::logic_error("remap_element<ShapeTet, ShapeQuad> not implemented!");
  }

  /** remap_element() (Tet, Tri) */
  template <int32 IP, int32 MP, int32 OP, int32 ncomp>
  DRAY_EXEC void remap_element(const ShapeTet,
                               OrderPolicy<IP> in_order_p,
                               const ReadDofPtr<Vec<Float, ncomp>> &in_field_rdp,
                               const ShapeTri,
                               OrderPolicy<MP> mesh_order_p,
                               const ReadDofPtr<Vec<Float, 3>> &mesh_rdp,
                               OrderPolicy<OP> out_order_p,
                               WriteDofPtr<Vec<Float, ncomp>> out_field_wdp)
  {
        // TODO this is where we should evaluate instead of merely lookup.
    throw std::logic_error("remap_element<ShapeTet, ShapeTri> not implemented!");
  }



  template <typename OutShape, int32 MP, class FElemT>
  std::shared_ptr<FieldBase> ReMapField_execute(const LocationSet &location_set,
                                                OutShape out_shape,
                                                OrderPolicy<MP> _mesh_order_p,
                                                Field<FElemT> &in_field)
  {
    // The output field type is based on the input field type,
    // but can also depend on the mesh order policy.

    const OrderPolicy<MP> mesh_order_p = _mesh_order_p;

    using InShape = typename AdaptGetShape<FElemT>::type;
    using InOrderPolicy = typename AdaptGetOrderPolicy<FElemT>::type;
    const InOrderPolicy in_order_p = adapt_get_order_policy(FElemT(), in_field.order());

    // TODO Evaluate (in Lagrange) on the surface to find out-field dof ref coords.
    // Then the output field type can match the input field order policy.

    ///using OutOrderPolicy = typename AdaptGetOrderPolicy<FElemT>::type;
    using OutOrderPolicy = OrderPolicy<MP>;

    ///const OutOrderPolicy out_order_p = adapt_get_order_policy(FElemT(), field.order());
    const OutOrderPolicy out_order_p = mesh_order_p;

    const int32 out_order = eattr::get_order(out_order_p);
    const int32 out_npe = eattr::get_num_dofs(out_shape, out_order_p);

    constexpr int32 ncomp = FElemT::get_ncomp();

    using OutFElemT = Element<2,
                              ncomp,
                              eattr::get_etype(out_shape),
                              eattr::get_policy_id(OutOrderPolicy())>;

    // Inputs.
    DeviceField<FElemT> device_in_field(in_field);
    DeviceGridFunctionConst<3> device_rcoords(location_set.m_rcoords);
    const int32 *host_cell_id_ptr = location_set.m_host_cell_id.get_device_ptr_const();

    const int32 num_out_elems = location_set.m_rcoords.get_num_elem();

    // Output.
    GridFunction<ncomp> out_field_gf;
    out_field_gf.resize_counting(num_out_elems, out_npe);
    DeviceGridFunction<ncomp> out_field_dgf(out_field_gf);

    RAJA::forall<for_policy>(RAJA::RangeSegment(0, num_out_elems), [=] DRAY_LAMBDA (int32 oeid)
    {
        const int32 host_cell_id = host_cell_id_ptr[oeid];
        FElemT in_felem = device_in_field.get_elem(host_cell_id);
        const ReadDofPtr<Vec<Float, ncomp>> in_field_rdp = in_felem.read_dof_ptr();
        const ReadDofPtr<Vec<Float, 3>> mesh_rdp = device_rcoords.get_rdp(oeid);
        WriteDofPtr<Vec<Float, ncomp>> out_field_wdp = out_field_dgf.get_wdp(oeid);

        remap_element(InShape(),
                      in_order_p,
                      in_field_rdp,
                      OutShape(),
                      mesh_order_p,
                      mesh_rdp,
                      out_order_p,
                      out_field_wdp);
    });

    return std::make_shared<Field<OutFElemT>>(out_field_gf, out_order, in_field.name());
  }



  // ExtractIsosurfaceFunctor
  struct ExtractIsosurfaceFunctor
  {
    Float m_iso_value;
    DataSet *m_input_dataset;

    DataSet m_output_tris;
    DataSet m_output_quads;

    ExtractIsosurfaceFunctor(Float iso_value, DataSet *input_dataset)
      : m_iso_value(iso_value),
        m_input_dataset(input_dataset)
    { }

    template <typename TopologyT, typename FieldT>
    void operator()(TopologyT &topo, FieldT &field)
    {
      auto output = ExtractIsosurface_execute(topo,
                                              field,
                                              m_iso_value,
                                              m_input_dataset);
      m_output_tris = output.first;
      m_output_quads = output.second;
    }
  };


  // execute() wrapper
  std::pair<DataSet, DataSet> ExtractIsosurface::execute(DataSet &data_set)
  {
    // Extract isosurface mesh.
    ExtractIsosurfaceFunctor func(m_iso_value, &data_set);
    dispatch_3d(data_set.topology(), data_set.field(m_iso_field_name), func);

    // Return dataset.
    return {func.m_output_tris, func.m_output_quads};
  }

  std::pair<Collection, Collection> ExtractIsosurface::execute(Collection &collxn)
  {
    Collection out_collxn_first;
    Collection out_collxn_second;
    for (DataSet ds : collxn.domains())
    {
      std::pair<DataSet, DataSet> ds_pair = this->execute(ds);
      out_collxn_first.add_domain(ds_pair.first);
      out_collxn_second.add_domain(ds_pair.second);
    }
    return {out_collxn_first, out_collxn_second};
  }


}
