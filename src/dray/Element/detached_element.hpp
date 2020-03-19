// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_DETACHED_ELEMENT_HPP
#define DRAY_DETACHED_ELEMENT_HPP

#include <dray/types.hpp>
#include <dray/Element/dof_access.hpp>

namespace dray
{
  /**
   * @brief Provide on-the-fly storage for a new element using new/delete.
   *
   * Usage:
   *     {
   *       using ElemT = Element<2, 3, ElemType::Tri, Order::General>;
   *       const int32 order = 3;
   *       DetachedElement temp_elem_storage(ElemT{}, order);
   *       WriteDofPtr<Vec<Float, 3>> writeable{temp_elem_storage.get_write_dof_ptr()};
   *
   *       external_old_element.split_into(writeable);
   *
   *       ElemT free_elem(writeable.to_readonly_dof_ptr(), order);
   *       AABB<3> sub_bounds = free_elem.get_bounds()
   *
   *       //...
   *       // When temp_elem_storage goes out of scope, the memory is delete'd.
   *     }
   */
  template <int32 ncomp>
  class DetachedElement
  {
    private:
      int32 *m_ctrl_idx;
      Vec<Float, ncomp> *m_values;
      int32 m_el_dofs;

    public:
      // DetachedElement()
      DRAY_EXEC DetachedElement() :
        m_ctrl_idx(nullptr),
        m_values(nullptr),
        m_el_dofs(0)
      {}

      // DetachedElement<ElemT>( , order)
      //
      // Construct a new DetachedElement with enough storage for ElemT.
      // Do not modify elem, just construct DetachedElement.
      template <class ElemT>
      DRAY_EXEC explicit DetachedElement(const ElemT, int32 order)
      {
        m_el_dofs = ElemT::get_num_dofs(order);
        m_ctrl_idx = new int32[m_el_dofs];
        m_values = new Vec<Float, ncomp>[m_el_dofs];
      }

      // destroy()
      DRAY_EXEC void destroy()
      {
        if (m_ctrl_idx != nullptr)
          delete [] m_ctrl_idx;
        if (m_values != nullptr)
          delete [] m_values;
      }

      // ~DetachedElement()
      DRAY_EXEC ~DetachedElement()
      {
        destroy();
      }

      // resize_to()
      template <class ElemT>
      DRAY_EXEC void resize_to(const ElemT, int32 order)
      {
        const int32 num_dofs = ElemT::get_num_dofs(order);
        if (m_el_dofs == num_dofs)
          return;

        destroy();

        m_el_dofs = num_dofs;
        m_ctrl_idx = new int32[m_el_dofs];
        m_values = new Vec<Float, ncomp>[m_el_dofs];
      }

      // get_write_dof_ptr()
      //
      // Use this after the 'order' constructor or 'resize_to'.
      DRAY_EXEC WriteDofPtr<Vec<Float, ncomp>> get_write_dof_ptr()
      {
        WriteDofPtr<Vec<Float, ncomp>> w_dof_ptr;
        w_dof_ptr.m_offset_ptr = m_ctrl_idx;
        w_dof_ptr.m_dof_ptr = m_values;
        return w_dof_ptr;
      }
  };

}//namespace dray

#endif//DRAY_DETACHED_ELEMENT_HPP
