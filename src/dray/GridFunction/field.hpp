// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_FIELD_HPP
#define DRAY_FIELD_HPP

#include <dray/Element/element.hpp>
#include <dray/GridFunction/grid_function.hpp>
#include <dray/GridFunction/field_base.hpp>
#include <dray/exports.hpp>
#include <dray/vec.hpp>

namespace dray
{

template <uint32 dim, uint32 ncomp, ElemType etype, Order P>
using FieldElem = Element<dim, ncomp, etype, P>;


template <class ElemT, uint32 ncomp> struct FieldOn_
{
  using get_type =
  Element<ElemT::get_dim (), ncomp, ElemT::get_etype (), ElemT::get_P ()>;
};


//
// FieldOn<>
//
// Get element type that is the same element type but different number of
// components. E.g., make a scalar element type over a given mesh element type:
//    using MeshElemT = MeshElem<float32, 3u, Quad, General>;
//    using FieldElemT = FieldOn<MeshElemT, 1u>;
template <class ElemT, uint32 ncomp>
using FieldOn = typename FieldOn_<ElemT, ncomp>::get_type;

// forward declare so we can have template friend
template <typename ElemT> class DeviceField;
/*
 * @class Field
 * @brief Host-side access to a collection of elements (just knows about the geometry, not fields).
 */
template <class ElemT> class Field : public FieldBase
{
  protected:
  GridFunction<ElemT::get_ncomp ()> m_dof_data;
  int32 m_poly_order;
  std::vector<Range> m_ranges;

  public:
  Field () = delete; // For now, probably need later.
  Field (const GridFunction<ElemT::get_ncomp ()>
         &dof_data, int32 poly_order,
         const std::string name = "");

  friend class DeviceField<ElemT>;

  virtual int32 order() const override;

  int32 get_poly_order () const
  {
    return m_poly_order;
  }

  int32 get_num_elem () const
  {
    return m_dof_data.get_num_elem ();
  }

  // TODO should this be removed?
  GridFunction<ElemT::get_ncomp ()> get_dof_data ()
  {
    return m_dof_data;
  }

  virtual std::vector<Range> range () const override;

  virtual std::string type_name() const override;

};

// Element<topo dims, ncomps, base_shape, polynomial order>
using HexScalar  = Element<3u, 1u, ElemType::Quad, Order::General>;
using HexVector = Element<3u, 3u, ElemType::Quad, Order::General>;
using QuadScalar = Element<2u, 1u,ElemType::Quad, Order::General>;
using QuadVector = Element<2u, 3u,ElemType::Quad, Order::General>;
} // namespace dray
#endif // DRAY_FIELD_HPP
