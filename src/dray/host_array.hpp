// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_HOST_ARRAY
#define DRAY_HOST_ARRAY

#include <dray/array.hpp>
#include <dray/types.hpp>
#include <dray/exports.hpp>

namespace dray
{

template <typename T>
class ConstHostArray
{
  public:
    ConstHostArray() = delete;
    ConstHostArray(Array<T> array)
      : m_size(array.size()),
        m_ncomp(array.ncomp()),
        m_host_ptr(array.get_host_ptr_const())
    {}

    DRAY_EXEC size_t size() const { return m_size; }
    DRAY_EXEC int32 ncomp() const { return m_ncomp; }

    DRAY_EXEC const T & get_item(size_t item_idx, int32 component = 0) const
    {
      return m_host_ptr[item_idx * m_ncomp + component];
    }

  protected:
    size_t m_size;
    int32 m_ncomp;
    const T * m_host_ptr;
};


template <typename T>
class NonConstHostArray
{
  public:
    NonConstHostArray() = delete;
    NonConstHostArray(Array<T> array)
      : m_size(array.size()),
        m_ncomp(array.ncomp()),
        m_host_ptr(array.get_host_ptr())
    {}

    DRAY_EXEC size_t size() const { return m_size; }
    DRAY_EXEC int32 ncomp() const { return m_ncomp; }

    DRAY_EXEC T & get_item(size_t item_idx, int32 component = 0) const
    {
      return m_host_ptr[item_idx * m_ncomp + component];
    }

  protected:
    size_t m_size;
    int32 m_ncomp;
    T * m_host_ptr;
};

}

#endif//DRAY_HOST_ARRAY
