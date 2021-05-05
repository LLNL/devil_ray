// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_ARRAY_MAPPING_HPP
#define DRAY_ARRAY_MAPPING_HPP

#include <memory>

#include <conduit.hpp>

#include <dray/types.hpp>
#include <dray/array.hpp>

namespace dray
{
  namespace detail
  {
    class WindowBlock;

    // ==================================
    // ArrayMapping
    // ==================================
    class ArrayMapping
    {
      public:
        void active_domain(int32 domain_index);
        int32 active_domain() const;

        // register the field first.
        const WindowBlock & register_uniform_low_order_field(
            const std::string &field_name,
            const conduit::Node &n_field);

        // retrieve the field later if it was registered before.
        const WindowBlock & retrieve_uniform_low_order_field(
            const std::string &field_name) const;

        bool field_exists(int32 domain_index, const std::string &field_name) const;

      protected:
        using DomainIndexT = int32;
        using FieldNameT = std::string;

        int32 m_active_domain = 0;
        std::map<DomainIndexT, std::map<FieldNameT, std::unique_ptr<WindowBlock>>> m_domains;
    };


    //
    // WindowBlockMeta
    //
    struct WindowBlockMeta
    {
      WindowBlockMeta() = default;

      // registration subtask
      WindowBlockMeta(const std::string &field_name,
                      const conduit::Node &n_field);

      size_t num_moments() const { return m_num_moments; }

      int32 m_num_axes;
      std::vector<size_t> m_total_origin;
      std::vector<size_t> m_total_upper;
      std::vector<size_t> m_total_shape;
      unsigned long long m_total_size;

      size_t m_num_components;
      size_t m_num_items;
      size_t m_num_moments;

      std::vector<int> m_dray_2_conduit_axis_ordering;

      std::vector<size_t> m_dray_strides;     // also indexed by conduit axis
      int m_component_axis;
    };


    //
    // WindowMeta
    //
    struct WindowMeta
    {
      WindowMeta() = default;

      // registration subtask
      WindowMeta(const std::string &field_name,
                 const conduit::Node &n_window,
                 const WindowBlockMeta &block_meta);

      // import
      template <typename ValT>
      void copy_conduit_2_dray(const WindowBlockMeta &block,
                               const conduit::Node &n_window_data,
                               Array<Float> all_windows) const;

      // export
      template <typename ValT>
      void copy_dray_2_conduit(const WindowBlockMeta &block,
                               const Array<Float> &all_windows,
                               conduit::Node &n_window_data) const;

      std::vector<size_t> m_conduit_strides;  // indexed by conduit axis
      std::vector<size_t> m_conduit_shape;
      size_t m_item_offset;
      size_t m_component_offset;
    };


    //
    // WindowBlock
    //
    class WindowBlock
    {
      public:
        WindowBlock() = default;

        // registration subtask
        WindowBlock(const std::string &field_name,
                    const conduit::Node &n_field);

        size_t num_moments() const { return m_meta.num_moments(); }

        // pre-import
        void resize_dray_array(Array<Float> &all_windows) const
        {
          all_windows.resize(m_meta.m_num_items, m_meta.m_num_components);
        }

        // import
        void copy_conduit_2_dray(const conduit::Node &n_field,
                                 Array<Float> all_windows) const;

        // export
        void copy_dray_2_conduit(const Array<Float> &all_windows,
                                 conduit::Node &n_field) const;

      protected:
        std::vector<WindowMeta> m_windows;
        WindowBlockMeta m_meta;
    };


  }//namespace detail
}//namespace dray

#endif//DRAY_ARRAY_MAPPING_HPP
