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
#include <dray/error.hpp>

namespace dray
{

template <int32 dim, int32 ncomp, ElemType etype, int32 P_Order>
using FieldElem = Element<dim, ncomp, etype, P_Order>;

// forward declare so we can have template friend
template <typename ElemT> struct DeviceField;
template <typename ElemT> class Field;

/**
 * @class FieldFriend
 * @brief A mutual friend of all Field template class instantiations.
 * @note Avoids making all Field template instantiations friends of each other
 *       as well as friends to impostor 'Field' instantiations.
 */
class FieldFriend
{
  /**
   * @brief Use a fast path based on field order, or go back to general path.
   * @tparam new_order should equal the field order, or be -1.
   */
  public:
    template <class ElemT, int new_order>
    static
    Field<FieldElem<ElemT::get_dim(), ElemT::get_ncomp(), ElemT::get_etype(), new_order>>
    to_fixed_order(Field<ElemT> &in_field);
};




/*
 * @class Field
 * @brief Host-side access to a collection of elements (just knows about the geometry, not fields).
 */
template <class ElemT> class Field : public FieldBase
{
  protected:
  GridFunction<ElemT::get_ncomp ()> m_dof_data;
  int32 m_poly_order;
  mutable bool m_range_calculated;
  mutable std::vector<Range> m_ranges;

  // Accept input data (as shared).
  // Useful for keeping same data but changing class template arguments.
  // If kept protected, can only be called by Field<ElemT> or friends of Field<ElemT>.
  Field(const FieldBase &other_fb,
        GridFunction<ElemT::get_ncomp()> dof_data,
        int32 poly_order,
        bool range_calculated,
        std::vector<Range> ranges);

  public:
  Field () = delete; // For now, probably need later.
  Field (const GridFunction<ElemT::get_ncomp ()>
         &dof_data, int32 poly_order,
         const std::string name = "");

  friend struct DeviceField<ElemT>;
  friend FieldFriend;

  Field(Field &&other);
  Field(const Field &other);

  /**
   * @brief Use a fast path based on mesh order, or go back to general path.
   * @tparam new_order should equal the mesh order, or be -1.
   */
  template <int new_order>
  Field<FieldElem<ElemT::get_dim(), ElemT::get_ncomp(), ElemT::get_etype(), new_order>>
  to_fixed_order()
  {
    return FieldFriend::template to_fixed_order<ElemT, new_order>(*this);
  }

  virtual void to_node(conduit::Node &n_field) override;

  virtual int32 order() const override;

  virtual void eval(const Array<Location> locs, Array<Float> &values) override;

  int32 get_poly_order () const
  {
    return m_poly_order;
  }

  int32 get_num_elem () const
  {
    return m_dof_data.get_num_elem ();
  }

  GridFunction<ElemT::get_ncomp ()> get_dof_data ()
  {
    return m_dof_data;
  }

  const GridFunction<ElemT::get_ncomp ()> & get_dof_data () const
  {
    return m_dof_data;
  }

  virtual std::vector<Range> range () const override;

  virtual std::string type_name() const override;

  static Field uniform_field(int32 num_els,
                             const Vec<Float, ElemT::get_ncomp()> &val,
                             const std::string &name = "");
};


// FieldFriend::to_fixed_order()
//   Could go in a .tcc file.
//
template <class ElemT, int new_order>
Field<FieldElem<ElemT::get_dim(), ElemT::get_ncomp(), ElemT::get_etype(), new_order>>
FieldFriend::to_fixed_order(Field<ElemT> &in_field)
{
  // Finite set of supported cases. Initially (bi/tri)quadrtic and (bi/tri)linear.
  static_assert(
      (new_order == -1 || new_order == 1 || new_order == 2),
      "Using fixed order 'new_order' not supported.\n"
      "Make sure Element<> for that order is instantiated "
      "and FieldFriend::to_fixed_order() "
      "is updated to include existing instantiations");

  if (!(new_order == -1 || new_order == in_field.get_poly_order()))
  {
    std::stringstream msg_ss;
    msg_ss << "Requested new_order (" << new_order
           << ") does not match existing poly order (" << in_field.get_poly_order()
           << ").";
    const std::string msg{msg_ss.str()};
    DRAY_ERROR(msg);
  }

  using NewElemT = FieldElem<ElemT::get_dim(), ElemT::get_ncomp(), ElemT::get_etype(), new_order>;

  return Field<NewElemT>(in_field,
                         in_field.m_dof_data,
                         in_field.m_poly_order,
                         in_field.m_range_calculated,
                         in_field.m_ranges);
}



// Element<topo dims, ncomps, base_shape, polynomial order>
using HexScalar  = Element<3u, 1u, ElemType::Tensor, Order::General>;
using HexScalar_P0  = Element<3u, 1u, ElemType::Tensor, Order::Constant>;
using HexScalar_P1  = Element<3u, 1u, ElemType::Tensor, Order::Linear>;
using HexScalar_P2  = Element<3u, 1u, ElemType::Tensor, Order::Quadratic>;

using TetScalar  = Element<3u, 1u, ElemType::Simplex, Order::General>;
using TetScalar_P0 = Element<3u, 1u, ElemType::Simplex, Order::Constant>;
using TetScalar_P1 = Element<3u, 1u, ElemType::Simplex, Order::Linear>;
using TetScalar_P2 = Element<3u, 1u, ElemType::Simplex, Order::Quadratic>;

using QuadScalar  = Element<2u, 1u, ElemType::Tensor, Order::General>;
using QuadScalar_P0 = Element<2u, 1u, ElemType::Tensor, Order::Constant>;
using QuadScalar_P1 = Element<2u, 1u, ElemType::Tensor, Order::Linear>;
using QuadScalar_P2 = Element<2u, 1u, ElemType::Tensor, Order::Quadratic>;

using TriScalar  = Element<2u, 1u, ElemType::Simplex, Order::General>;
using TriScalar_P0 = Element<2u, 1u, ElemType::Simplex, Order::Constant>;
using TriScalar_P1 = Element<2u, 1u, ElemType::Simplex, Order::Linear>;
using TriScalar_P2 = Element<2u, 1u, ElemType::Simplex, Order::Quadratic>;


using HexVector = Element<3u, 3u, ElemType::Tensor, Order::General>;
using QuadVector = Element<2u, 3u,ElemType::Tensor, Order::General>;


} // namespace dray
#endif // DRAY_FIELD_HPP
