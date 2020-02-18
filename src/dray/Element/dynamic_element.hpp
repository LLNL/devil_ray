// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_DYNAMIC_ELEMENT_HPP
#define DRAY_DYNAMIC_ELEMENT_HPP

#include <dray/types.hpp>
#include <dray/vec.hpp>
#include <dray/Element/element.hpp>
#include <dray/Element/dof_access.hpp>

namespace dray
{

template <typename ElemT>
class DynamicElement
{
  static constexpr uint32 dim = ElemT::get_dim();
  static constexpr uint32 ncomp = ElemT::get_ncomp();
  static constexpr ElemType etype = ElemT::get_etype();

  protected:
    WriteDofPtr<Vec<Float, ncomp>> m_dof_ptr;

    int32 m_order;
    int32 m_size;  // Number of Vec<Float, ncomp>

  public:
    DynamicElement();
   ~DynamicElement();
    DynamicElement(int32 order);
    DynamicElement(ReadDofPtr<Vec<Float, ncomp>> dof_ptr, int32 order);
    /// DynamicElement(const ElemT &other);            // Deep copy constructor.
    DynamicElement(const DynamicElement &other);   // Deep copy constructor.
    DynamicElement(DynamicElement &&other);        // Move constructor.
    DynamicElement & operator=(const DynamicElement &other);    // Deep copy assignment.
    DynamicElement & operator=(DynamicElement &&other);         // Move assignment.

    void order(int32 new_order);   // invalidates references.

    int32 order() const { return m_order; }
    int32 size() const { return m_size; }

    WriteDofPtr<Vec<Float, ncomp>> write_dof_ptr() { return m_dof_ptr; }

    ElemT get_elem() const;

    static constexpr uint32 get_dim() { return dim; }
    static constexpr uint32 get_ncomp() { return ncomp; }
    static constexpr ElemType get_etype() { return etype; }
};


// Default constructor.
template <typename ElemT>
DynamicElement<ElemT>::DynamicElement() :
  m_dof_ptr{nullptr, nullptr},
  m_order{0},
  m_size{0}
{
}

// Destructor.
template <typename ElemT>
DynamicElement<ElemT>::~DynamicElement()
{
  if (m_dof_ptr.m_offset_ptr != nullptr)
    delete [] m_dof_ptr.m_offset_ptr;
  if (m_dof_ptr.m_dof_ptr != nullptr)
    delete [] m_dof_ptr.m_dof_ptr;
}

// Constructor(order).
template <typename ElemT>
DynamicElement<ElemT>::DynamicElement(int32 order) :
  m_dof_ptr{nullptr, nullptr},
  m_order{order},
  m_size{ElemT::get_num_dofs(order)}
{
  int32 * counting_array = new int32[m_size];
  for (int ii = 0; ii < m_size; ++ii)
    counting_array[ii] = ii;

  m_dof_ptr.m_offset_ptr = counting_array;
  m_dof_ptr.m_dof_ptr = new Vec<Float, ncomp> [m_size];
}

// Constructor(dof_ptr, order)
template <typename ElemT>
DynamicElement<ElemT>::DynamicElement(
    ReadDofPtr<Vec<Float, ncomp>> dof_ptr,
    int32 order)
  :
  DynamicElement(order)
{
  // By delegating to other constructor, memory allocated. Now, have to copy.
  for (int ii = 0; ii < m_size; ++ii)
    m_dof_ptr.m_dof_ptr[ii] = dof_ptr[ii];
}

/// // Constructor(Element)
/// template <typename ElemT>
/// DynamicElement<ElemT>::DynamicElement(const ElemT &other) :
///   DynamicElement(other.get_order())
/// {
///   // By delegating to other constructor, memory allocated. Now, have to copy.
///   for (int ii = 0; ii < m_size; ++ii)
///     m_dof_ptr.m_dof_ptr[ii] = other.m_dof_ptr[ii];  // Violates encapsulation of Element
/// }

// Constructor(DynamicElement)
template <typename ElemT>
DynamicElement<ElemT>::DynamicElement(const DynamicElement &other) :
  DynamicElement(other.order())
{
  // By delegating to other constructor, memory allocated. Now, have to copy.
  for (int ii = 0; ii < m_size; ++ii)
    m_dof_ptr.m_dof_ptr[ii] = other.m_dof_ptr.m_dof_ptr[ii];
  // TODO if there is an Umpire interface for device memcpy, use it.
}

// Move constructor.
template <typename ElemT>
DynamicElement<ElemT>::DynamicElement(DynamicElement &&other) :
  m_dof_ptr{other.m_dof_ptr.m_offset_ptr, other.m_dof_ptr.m_dof_ptr},
  m_order{other.m_order},
  m_size{other.m_size}
{
  other.m_dof_ptr.m_offset_ptr = nullptr;
  other.m_dof_ptr.m_dof_ptr = nullptr;
  other.m_order = 0;
  other.m_size = 0;
}

// Copy assignment.
template <typename ElemT>
DynamicElement<ElemT> & DynamicElement<ElemT>::operator=(const DynamicElement &other)
{
  if (m_order != other.order())
    order(other.order());

  // Memory allocated. Now, have to copy.
  for (int ii = 0; ii < m_size; ++ii)
    m_dof_ptr.m_dof_ptr[ii] = other.m_dof_ptr.m_dof_ptr[ii];

  return *this;
}

// Move assignment.
template <typename ElemT>
DynamicElement<ElemT> & DynamicElement<ElemT>::operator=(DynamicElement &&other)
{
  m_dof_ptr = other.m_dof_ptr;
  m_order = other.m_order;
  m_size = other.m_size;

  other.m_dof_ptr.m_offset_ptr = nullptr;
  other.m_dof_ptr.m_dof_ptr = nullptr;
  other.m_order = 0;
  other.m_size = 0;
}

// order()
// @brief: Change order. Invalidates references.
template <typename ElemT>
void DynamicElement<ElemT>::order(int32 new_order)
{
  if (m_dof_ptr.m_offset_ptr != nullptr)
    delete [] m_dof_ptr.m_offset_ptr;
  if (m_dof_ptr.m_dof_ptr != nullptr)
    delete [] m_dof_ptr.m_dof_ptr;

  m_order = new_order;
  m_size = ElemT::get_num_dofs(new_order);

  int32 * counting_array = new int32[m_size];
  for (int ii = 0; ii < m_size; ++ii)
    counting_array[ii] = ii;

  m_dof_ptr.m_offset_ptr = counting_array;
  m_dof_ptr.m_dof_ptr = new Vec<Float, ncomp> [m_size];
}

// get_elem()
template <typename ElemT>
ElemT DynamicElement<ElemT>::get_elem() const
{
  ElemT element;
  const int32 no_el_id = -1;
  element.construct(no_el_id, m_dof_ptr.to_readonly_dof_ptr(), m_order);
  return element;
}

}//namespace dray


#endif//DRAY_DYNAMIC_ELEMENT_HPP
