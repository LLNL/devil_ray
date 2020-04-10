// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <dray/filters/isosurfacing.hpp>
#include <dray/error.hpp>
#include <dray/dispatcher.hpp>

#include <RAJA/RAJA.hpp>
#include <dray/policies.hpp>
#include <dray/exports.hpp>

#include <dray/derived_topology.hpp>
#include <dray/GridFunction/field.hpp>
#include <dray/GridFunction/device_field.hpp>
#include <dray/Element/elem_attr.hpp>
#include <dray/Element/iso_ops.hpp>
#include <dray/Element/detached_element.hpp>

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

  //
  // execute(topo, field)
  //
  template <class MElemT, class FElemT>
  DataSet ExtractIsosurface_execute(DerivedTopology<MElemT> &topo,
                                    Field<FElemT> &field,
                                    Float iso_value)
  {
    static_assert(FElemT::get_ncomp() == 1, "Can't take isosurface of a vector field");

    const Float isoval = iso_value;  // Local for capture
    const int32 n_el_in = field.get_num_elem();
    DeviceField<FElemT> dfield(field);

    constexpr int32 budget = 10;
    Array<uint8> count_subelem_required;
    Array<uint8> budget_maxed;

    RAJA::forall<for_policy>(RAJA::RangeSegment(0, n_el_in), [=] DRAY_LAMBDA (int32 eidx) {
        using eops::measure_isocut;
        using eops::IsocutInfo;

        FElemT felem_in = dfield.get_elem(eidx);

        const auto shape = adapt_get_shape(felem_in);
        constexpr ElemType etype = eattr::get_etype(shape);
        const auto order_p = dfield.get_order_policy();
        const int32 p = eattr::get_order(order_p);

        // Breadth-first subdivision (no priorities)
        DetachedElement<1> felem_store[budget];
        /// DetachedElement<3> melem_store[budget];
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

        using detail::stack_push;
        using detail::stack_pop;
        using detail::circ_enqueue;
        using detail::circ_dequeue;
        using detail::circ_queue_rotate;

        // Enqueue the input element.
        circ_enqueue(q, budget, q_begin, q_sz,  stack_pop(pool, pool_sz));
        felem_store[ q[q_begin] ].resize_to(shape, order_p);
        felem_store[ q[q_begin] ].populate_from(felem_in.read_dof_ptr());
        info_store[ q[q_begin] ] = measure_isocut(shape,
            felem_store[ q[q_begin] ].get_write_dof_ptr().to_readonly_dof_ptr(), isoval, p);

        // Test each element in the queue.
        //   If no cut, discard and return slot from store to pool.
        //   If simple cut, keep slot reserved in store.
        //   If non simple cut, perform a split if there is room to do so.
        while ( q_sz > 0 )
        {
          ceiling = (budget-pool_sz > ceiling ? budget-pool_sz : ceiling);

          const IsocutInfo &head_info = info_store[ q[q_begin] ];
          if (head_info.m_cut_type_flag == 0)
          {
            std::cout << "[" << eidx << "] No cut.\n";
            stack_push(pool, pool_sz,  circ_dequeue(q, budget, q_begin, q_sz));  // restore to circulation.

            stuck = false;
            stuck_counter = 0;
          }
          else if (head_info.m_cut_type_flag < 8)
          {
            if (head_info.m_cut_type_flag & IsocutInfo::CutSimpleTri)
              std::cout << "[" << eidx << "] Simple tri\n";
            else if (head_info.m_cut_type_flag & IsocutInfo::CutSimpleQuad)
              std::cout << "[" << eidx << "] Simple quad\n";

            circ_dequeue(q, budget, q_begin, q_sz);  // Reserve and remove from circulation.

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

              felem_store[daughter].resize_to(shape, order_p);
              felem_store[daughter].populate_from( felem_store[mother].get_write_dof_ptr().to_readonly_dof_ptr() );

              // Get the split direction.
              Split<etype> binary_split = pick_iso_simple_split(shape, info_store[mother]);
              std::cout << "[" << eidx << "] Splitting... " << binary_split << "\n";

              // Perform the split.
              split_inplace(shape, order_p, felem_store[mother].get_write_dof_ptr(), binary_split);
              split_inplace(shape, order_p, felem_store[daughter].get_write_dof_ptr(), binary_split.get_complement());

              // Update isocut info.
              info_store[mother] = measure_isocut(shape,
                  felem_store[mother].get_write_dof_ptr().to_readonly_dof_ptr(), isoval, p);
              info_store[daughter] = measure_isocut(shape,
                  felem_store[daughter].get_write_dof_ptr().to_readonly_dof_ptr(), isoval, p);

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
                std::cout << "  ..stuck (" << int32(stuck_counter) << ")\n";
              }
              else  // Give up
              {
                std::cout << "Giving up!\n";
                break;
              }
            }
          }
        }

        std::cout << "[" << eidx << "]*Finished subdividing, "
                  << "circulation (" << int32(ceiling) << "/" << int32(budget) << "),  "
                  << "final ("
                  << int32(budget-pool_sz) << "/" << int32(budget) << ")"
                  << (stuck ? "(+)" : "")
                  << "*\n";

    });

    DRAY_ERROR("Implementation of ExtractIsosurface_execute() not done yet");
  }


  // ExtractIsosurfaceFunctor
  struct ExtractIsosurfaceFunctor
  {
    Float m_iso_value;

    DataSet m_output;

    ExtractIsosurfaceFunctor(Float iso_value)
      : m_iso_value{iso_value}
    { }

    template <typename TopologyT, typename FieldT>
    void operator()(TopologyT &topo, FieldT &field)
    {
      m_output = ExtractIsosurface_execute(topo, field, m_iso_value);
    }
  };

  // execute() wrapper
  DataSet ExtractIsosurface::execute(DataSet &data_set)
  {
    ExtractIsosurfaceFunctor func(m_iso_value);
    dispatch_3d(data_set.topology(), data_set.field(m_iso_field_name), func);
    return func.m_output;
  }

}
