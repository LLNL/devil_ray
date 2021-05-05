// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <dray/io/array_mapping.hpp>
#include <dray/host_array.hpp>
#include <dray/error.hpp>
#include <dray/utils/data_logger.hpp>
#include <dray/math.hpp>

#include <limits>

namespace dray
{
  namespace detail
  {

    using namespace conduit;

    template <typename T>
    static T extract(const conduit::Node & node, size_t index)
    {
      return static_cast<const T *>(node.value())[index];
    }

    static int64 extract_signed(const conduit::Node & node, size_t index)
    {
      if (node.dtype().is_int8())
        return extract<int8>(node, index);
      else if (node.dtype().is_int32())
        return extract<int32>(node, index);
      else if (node.dtype().is_int64())
        return extract<int64>(node, index);
      else
        throw std::logic_error("Called extract_signed() but node is not any of int8, int32, or int64.");
    }

    template <typename T>
    static T & extract_ref(conduit::Node & node, size_t index)
    {
      return static_cast<T *>(node.value())[index];
    }


    void ArrayMapping::active_domain(int32 domain_index)
    {
      m_active_domain = domain_index;
    }

    int32 ArrayMapping::active_domain() const
    {
      return m_active_domain;
    }

    bool ArrayMapping::field_exists(int32 domain_index,
                                    const std::string &field_name) const
    {
      const auto &domain = m_domains.find(domain_index);
      if (domain == m_domains.end())
        return false;
      else
        if (domain->second.find(field_name) == domain->second.end())
          return false;
        else
          return true;
    }

    // register_uniform_low_order_field()
    const WindowBlock & ArrayMapping::register_uniform_low_order_field(
        const std::string &field_name,
        const conduit::Node &n_field)
    {
      if (this->field_exists(m_active_domain, field_name))
      {
        DRAY_ERROR("ArrayMapping: Field '" << field_name \
                   << "' already exists in domain " << m_active_domain);
      }

      const int num_windows = n_field["values"].number_of_children();
      if (num_windows == 0)
      {
        DRAY_ERROR("Expected the values of field " \
                   << field_name \
                   << " to be formatted as one or more windows.");
      }

      // New window block.
      m_domains[m_active_domain][field_name]
          = std::unique_ptr<WindowBlock>(new WindowBlock(field_name, n_field));
      WindowBlock &window_block = *(m_domains[m_active_domain][field_name]);

      return window_block;
    }

    // retrieve_uniform_low_order_field()
    const WindowBlock & ArrayMapping::retrieve_uniform_low_order_field(
        const std::string &field_name) const
    {
      if (!this->field_exists(m_active_domain, field_name))
      {
        DRAY_ERROR("ArrayMapping: Field '" << field_name \
                   << "' requested in domain " << m_active_domain \
                   << ", but the field has not been registered.");
      }

      return *(m_domains.find(m_active_domain)->second.find(field_name)->second);
    }


    //
    // WindowBlock()  constructor
    //   Calls
    //   - WindowBlockMeta() constructor, and
    //   - WindowMeta() constructor
    //
    WindowBlock::WindowBlock(const std::string &field_name,
                             const conduit::Node &n_field)
    {
      const int num_windows = n_field["values"].number_of_children();
      const Node &n_values = n_field["values"];

      // Check for expected attributes in the 0th window.
      {
        std::stringstream ss;
        bool oops = false;
        bool complain = false;
        ss << "field " << field_name << "\n";
        if ((oops = !n_values.child(0).has_child("shape"), (complain |= oops, oops)))
          ss << "No shape!\n";
        if ((oops = !n_values.child(0).has_child("strides"), (complain |= oops, oops)))
          ss << "No strides!\n";
        if ((oops = !n_values.child(0).has_child("data"), (complain |= oops, oops)))
          ss << "No data!\n";
        if ((oops = !n_values.child(0).has_child("origin"), (complain |= oops, oops)))
          ss << "No origin!\n";
        if ((oops = !n_values.child(0).has_child("shape_labels"), (complain |= oops, oops)))
          ss << "No shape_labels!\n";
        if (complain)
          std::cerr << ss.str();
      }

      m_meta = WindowBlockMeta(field_name, n_field);

      // Add window-specific strides.
      for (int window = 0; window < num_windows; ++window)
        m_windows.emplace_back(field_name,
                               n_values.child(window),
                               m_meta);
    }

    //
    // WindowBlockMeta() constructor
    //
    WindowBlockMeta::WindowBlockMeta(const std::string &field_name,
                                     const conduit::Node &n_field)
    {
      // Assumes that the set of windows occupies
      // a contiguous multi-D rectangle in index space.

      const int num_windows = n_field["values"].number_of_children();
      const Node &n_values = n_field["values"];

      m_num_axes = n_values.child(0)["shape"].dtype().number_of_elements();
      m_total_origin.resize(m_num_axes,
                                       std::numeric_limits<size_t>::max());
      m_total_upper.resize(m_num_axes, 0);
      m_total_size = 0;
      m_total_shape.resize(m_num_axes, 0);

      // Calculate sizes.
      for (int window = 0; window < num_windows; ++window)
      {
        const Node &n_window = n_values.child(window);
        size_t window_size = 1;
        for (int axis = 0; axis < m_num_axes; ++axis)
        {
          const size_t next_origin = extract_signed(n_window["origin"], axis);
          const size_t next_shape = extract_signed(n_window["shape"], axis);
          const size_t next_upper = next_origin + next_shape;
          if (m_total_origin[axis] > next_origin)
            m_total_origin[axis] = next_origin;
          if (m_total_upper[axis] < next_upper)
            m_total_upper[axis] = next_upper;
          window_size *= next_shape;
        }
        m_total_size += window_size;
      }

      unsigned long long size_from_shape = 1;
      for (int axis = 0; axis < m_num_axes; ++axis)
      {
        m_total_shape[axis] = m_total_upper[axis] - m_total_origin[axis];
        size_from_shape *= m_total_shape[axis];
      }

      // Verify contiguous block.
      if (size_from_shape != m_total_size)
      {
        std::stringstream size_ss;
        size_ss << "Extents [";
        for (int axis = 0; axis < m_num_axes; ++axis)
          size_ss << m_total_origin[axis] << ", ";
        size_ss << "] -- [";
        for (int axis = 0; axis < m_num_axes; ++axis)
          size_ss << m_total_upper[axis] << ", ";
        size_ss << "] do not match the actual size (" << m_total_size << ")";
        throw std::logic_error(size_ss.str());
      }
      else
      {
        std::cout << "Field " << field_name
                  << ": Size (" << m_total_size
                  << ") is consistent with continuous block assumption.\n";
      }

      // "group" axis --> components
      // other axes --> items
      m_num_components = 1;
      m_num_items = 1;
      m_num_moments = 1;
      int group_axis = -1;
      int zone_axis = -1;
      for (int axis = 0; axis < m_num_axes; ++axis)
      {
        const std::string label =
            n_values.child(0)["shape_labels"][axis].as_string();
        std::cout << "Label: '"<< label << "'\n";

        if (label == "zone")
          zone_axis = axis;

        if (label == "moment")
          m_num_moments = m_total_shape[axis];  //TODO get legendre_order directly

        if (label == "group")
        {
          group_axis = axis;
          m_num_components = m_total_shape[axis];
        }
        else
        {
          m_num_items *= m_total_shape[axis];
        }
      }

      fprintf(stdout, "zone_axis==%d, group_axis==%d, "
                      "m_num_items==%lu, m_num_components==%lu\n",
                      zone_axis, group_axis,
                      m_num_items, m_num_components);

      // Zones should vary slowest.
      m_dray_2_conduit_axis_ordering.resize(m_num_axes, 0);
      int dray_axis = 0;
      for (int axis = 0; axis < m_num_axes; ++axis)
        if (axis != zone_axis)
          m_dray_2_conduit_axis_ordering[dray_axis++] = axis;
      if (zone_axis != -1)
        m_dray_2_conduit_axis_ordering[m_num_axes-1] = zone_axis;

      m_component_axis = group_axis;

      m_dray_strides.resize(m_num_axes, 0);

      // Destination strides for item index.
      // Component axis maps to dray::Array component,
      // so does not contribute to item index.
      size_t stride = 1;
      for (int dray_axis = 0; dray_axis < m_num_axes; ++dray_axis)
      {
        const int conduit_axis = m_dray_2_conduit_axis_ordering[dray_axis];
        if (conduit_axis != m_component_axis)
        {
          m_dray_strides[conduit_axis] = stride;
          stride *= m_total_shape[conduit_axis];
        }
      }
      if (m_component_axis != -1)
        m_dray_strides[m_component_axis] = 0;

      std::cout << "m_dray_strides: [";
      for (int axis = 0; axis < m_num_axes; ++axis)
        std::cout << m_dray_strides[axis] << ", ";
      std::cout << "]\n";
    }


    //
    // is_floating_point()
    //
    bool is_floating_point(const conduit::Node &values)
    {
      return values.dtype().is_float32()
             || values.dtype().is_float64();
    }



    //
    // WindowMeta() constructor
    //
    WindowMeta::WindowMeta(const std::string &field_name,
                           const conduit::Node &n_window,
                           const WindowBlockMeta &block)
    {
      m_item_offset = 0;
      m_component_offset = 0;

      const int num_axes = block.m_num_axes;
      m_conduit_shape.resize(num_axes);
      m_conduit_strides.resize(num_axes);

      for (int axis = 0; axis < num_axes; ++axis)
      {
        m_conduit_shape[axis] = extract_signed(n_window["shape"], axis);
        m_conduit_strides[axis] = extract_signed(n_window["strides"], axis);
        const size_t window_origin = extract_signed(n_window["origin"], axis);
        const size_t window_offset = window_origin - block.m_total_origin[axis];
        if (axis == block.m_component_axis)
          m_component_offset = window_offset;
        else
          m_item_offset += m_conduit_strides[axis] * window_offset;
      }

      if (!is_floating_point(n_window["data"]))
      {
        std::cout<<"non-floating point field '"<<field_name<<"'\n";
      }
    }


    // WindowBlock::copy_conduit_2_dray()   (import)
    void WindowBlock::copy_conduit_2_dray(
        const conduit::Node &n_field,
        Array<Float> all_windows) const
    {
      // Assume dray::Array has m_num_items items and m_num_components components.

      // Collect all windows in the field.
      int window_idx = 0;
      for (const WindowMeta &window : m_windows)
      {
        const Node &n_window = n_field["values"].child(window_idx++);

        if (n_window["data"].dtype().is_float32())
          window.copy_conduit_2_dray<float32>(m_meta,
                                              n_window["data"],
                                              all_windows);

        else if (n_window["data"].dtype().is_float64())
          window.copy_conduit_2_dray<float64>(m_meta,
                                              n_window["data"],
                                              all_windows);
      }
    }

    // WindowBlock::copy_dray_2_conduit()   (export)
    void WindowBlock::copy_dray_2_conduit(
        const Array<Float> &all_windows,
        conduit::Node &n_field) const
    {
      // Assume dray::Array has m_num_items items and m_num_components components.

      // Collect all windows in the field.
      int window_idx = 0;
      for (const WindowMeta &window : m_windows)
      {
        Node &n_window = n_field["values"].child(window_idx++);

        if (n_window["data"].dtype().is_float32())
          window.copy_dray_2_conduit<float32>(m_meta,
                                              all_windows,
                                              n_window["data"]);

        else if (n_window["data"].dtype().is_float64())
          window.copy_dray_2_conduit<float64>(m_meta,
                                              all_windows,
                                              n_window["data"]);
      }
    }


    //
    // MultiDigit
    //
    template <typename I = size_t>
    class MultiDigit
    {
      public:
        template <typename II>
        MultiDigit(const std::vector<II> &limits)
          : m_digits(limits.size(), 0),
            m_limits(limits.begin(), limits.end())
        {}

        void increment()
        {
          if (num_digits() == 0)
            return;

          m_digits[0]++;
          size_t carry_digit = 0;
          while (carry_digit < num_digits()
                 && m_digits[carry_digit] == m_limits[carry_digit])
          {
            m_digits[carry_digit] = 0;
            carry_digit++;
            if (carry_digit < num_digits())
              m_digits[carry_digit]++;
          }

          return;
        }

        size_t num_digits() const
        {
          return m_digits.size();
        }

        I digit(size_t i) const
        {
          return m_digits[i];
        }

        const std::vector<I> & digits() const
        {
          return m_digits;
        }

        const std::vector<I> & limits() const
        {
          return m_limits;
        }

        bool is_zero() const
        {
          bool digit_zero = true;
          for (size_t digit = 0; digit < num_digits(); ++digit)
            digit_zero &= (m_digits[digit] == 0);
          return digit_zero;
        }

      private:
        std::vector<I> m_digits;
        std::vector<I> m_limits;
    };






    // WindowMeta::copy_conduit_2_dray()   (import)
    template <typename ValT>
    void WindowMeta::copy_conduit_2_dray(const WindowBlockMeta &block,
                                         const conduit::Node &n_window_data,
                                         Array<Float> all_windows) const
    {
      NonConstHostArray<Float> all_windows_data(all_windows);
      MultiDigit<size_t> index(m_conduit_shape);
      const int num_axes = block.m_num_axes;
      const bool use_component = (block.m_component_axis != -1);
      do
      {
        size_t conduit_idx = 0;
        size_t dray_item_idx = 0;
        for (int axis = 0; axis < num_axes; ++axis)
        {
          conduit_idx += index.digit(axis) * m_conduit_strides[axis];
          dray_item_idx += index.digit(axis) * block.m_dray_strides[axis];
        }
        const size_t component =
            (use_component ?  index.digit(block.m_component_axis) : 0);

        all_windows_data.get_item(m_item_offset + dray_item_idx,
                                  m_component_offset + component)
            = extract<ValT>(n_window_data, conduit_idx);

        index.increment();
      }
      while (!index.is_zero());
    }

    // WindowMeta::copy_dray_2_conduit()   (export)
    template <typename ValT>
    void WindowMeta::copy_dray_2_conduit(const WindowBlockMeta &block,
                                         const Array<Float> &all_windows,
                                         conduit::Node &n_window_data) const
    {
      ConstHostArray<Float> all_windows_data(all_windows);
      MultiDigit<size_t> index(m_conduit_shape);
      const int num_axes = block.m_num_axes;
      const bool use_component = (block.m_component_axis != -1);
      do
      {
        size_t conduit_idx = 0;
        size_t dray_item_idx = 0;
        for (int axis = 0; axis < num_axes; ++axis)
        {
          conduit_idx += index.digit(axis) * m_conduit_strides[axis];
          dray_item_idx += index.digit(axis) * block.m_dray_strides[axis];
        }
        const size_t component =
            (use_component ?  index.digit(block.m_component_axis) : 0);

        if (isnan(all_windows_data.get_item(m_item_offset + dray_item_idx,
                                      m_component_offset + component)))
          throw std::logic_error("copy_dray_2_conduit() isnan");

        extract_ref<ValT>(n_window_data, conduit_idx)
          = all_windows_data.get_item(m_item_offset + dray_item_idx,
                                      m_component_offset + component);

        index.increment();
      }
      while (!index.is_zero());
    }


  }//namespace detail
}//namespace dray
