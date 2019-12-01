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
  Range<> m_range; // TODO aabb

  public:
  Field () = delete; // For now, probably need later.
  Field (const GridFunction<ElemT::get_ncomp ()>
         &dof_data, int32 poly_order,
         const std::string name = "");

  friend class DeviceField<ElemT>;

  virtual int32 order() const override;
  //
  // get_poly_order()
  int32 get_poly_order () const
  {
    return m_poly_order;
  }

  //
  // get_num_elem()
  int32 get_num_elem () const
  {
    return m_dof_data.get_num_elem ();
  }

  //
  // get_dof_data()  // TODO should this be removed?
  GridFunction<ElemT::get_ncomp ()> get_dof_data ()
  {
    return m_dof_data;
  }

  Range<> get_range () const; // TODO aabb

};

// Element<topo dims, ncomps, base_shape, polynomial order>
using Hex1  = Element<3u, 1u, ElemType::Quad, Order::General>;
using Quad1 = Element<2u, 1u,ElemType::Quad, Order::General>;
} // namespace dray
#endif // DRAY_FIELD_HPP
