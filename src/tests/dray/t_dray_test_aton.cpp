// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "t_utils.hpp"
#include "test_config.h"
#include "gtest/gtest.h"
#include <dray/filters/first_scatter.hpp>
#include <dray/io/blueprint_reader.hpp>
#include <conduit_blueprint.hpp>
#include <conduit_relay.hpp>

#include <dray/error.hpp>
#include <dray/GridFunction/low_order_field.hpp>
#include <dray/uniform_topology.hpp>
#include <dray/host_array.hpp>
#include <dray/utils/data_logger.hpp>

#include <limits>


namespace aton
{
namespace detail
{
  using namespace dray;

  // --------------------------------------------------------
  Collection import_into_uniform(const conduit::Node &n_all_domains);
  DataSet import_domain_into_uniform(const conduit::Node &n_dataset);
  void uniform_low_order_fields(const conduit::Node &n_dataset, DataSet &dataset);
  Array<Float> fill_array(const conduit::Node &values);
  bool is_floating_point(const conduit::Node &values);

  class ArrayFiller
  {
    public:
      ArrayFiller(int num_axes,
                  const std::vector<int> &axis_ordering,
                  const std::vector<size_t> &total_origin,
                  const std::vector<size_t> &total_shape,
                  int component_axis);

      template <typename shape_t>
      void fill_window(Array<Float> all_windows,
                       const conduit::Node &n_window_data,
                       const conduit::Node &n_window_shape,
                       const conduit::Node &n_window_strides,
                       const conduit::Node &n_window_origin);

      template <typename shape_t, typename ValT>
      void copy_values(Array<Float> all_windows,
                       const conduit::Node &n_window_data,
                       const std::vector<shape_t> shape,
                       const conduit::Node &n_window_strides,
                       size_t item_offset,
                       size_t component_offset);

    private:
      bool m_use_components;
      int m_component_axis;
      std::vector<size_t> m_total_origin;
      std::vector<size_t> m_dest_strides;
  };

  // --------------------------------------------------------
}
}


TEST(aton_dray, aton_import)
{
  conduit::Node data;
  conduit::relay::io::load("../../../debug/external_source.json", "json", data);

  dray::Collection dray_collection = aton::detail::import_into_uniform(data);
}



namespace aton
{
namespace detail
{
  using namespace dray;

  Collection import_into_uniform(const conduit::Node &n_all_domains)
  {
    const int doms = n_all_domains.number_of_children();

    if (doms == 0)
    {
      DRAY_ERROR("empty domain list");
    }

    Collection collection;
    for(int i = 0; i < doms; ++i)
    {
      const conduit::Node &domain = n_all_domains.child(i);
      int domain_id = 0;
      if(domain.has_path("state/domain_id"))
        domain_id = domain["state/domain_id"].to_int32();
      DRAY_INFO("Importing domain "<<domain_id);
      DataSet dset = import_domain_into_uniform(domain);
      collection.add_domain(dset);
    }

    return collection;
  }

  DataSet import_domain_into_uniform(const conduit::Node &n_dataset)
  {
    const int num_topos = n_dataset["topologies"].number_of_children();

    if(num_topos != 1)
    {
      DRAY_ERROR("Only a single topology is supported");
    }
    const conduit::Node &topo = n_dataset["topologies"].child(0);

    if(topo["type"].as_string() != "uniform")
    {
      DRAY_ERROR("Only uniform topology implemented");
    }
    const std::string cname = topo["coordset"].as_string();
    const conduit::Node coords = n_dataset["coordsets/"+cname];

    const Node &n_dims = coords["dims"];

    // cell dims
    int dims_i = n_dims["i"].to_int() - 1;
    int dims_j = n_dims["j"].to_int() - 1;
    int dims_k = 1;
    bool is_2d = true;

    // check for 3d
    if(n_dims.has_path("k"))
    {
      dims_k = n_dims["k"].to_int() - 1;
      is_2d = false;
    }

    Float origin_x = 0.0f;
    Float origin_y = 0.0f;
    Float origin_z = 0.0f;

    if(coords.has_path("origin"))
    {
      const Node &n_origin = coords["origin"];

      if(n_origin.has_child("x"))
      {
        origin_x = n_origin["x"].to_float32();
      }

      if(n_origin.has_child("y"))
      {
        origin_y = n_origin["y"].to_float32();
      }

      if(n_origin.has_child("z"))
      {
        origin_z = n_origin["z"].to_float32();
      }
    }

    Float spacing_x = 1.0f;
    Float spacing_y = 1.0f;
    Float spacing_z = 1.0f;

    if(coords.has_path("spacing"))
    {
      const Node &n_spacing = coords["spacing"];

      if(n_spacing.has_path("dx"))
      {
          spacing_x = n_spacing["dx"].to_float32();
      }

      if(n_spacing.has_path("dy"))
      {
          spacing_y = n_spacing["dy"].to_float32();
      }

      if(n_spacing.has_path("dz"))
      {
          spacing_z = n_spacing["dz"].to_float32();
      }
    }

    Vec<Float,3> spacing{spacing_x, spacing_y, spacing_z};
    Vec<Float,3> origin{origin_x, origin_y, origin_z};
    Vec<int32,3> dims{dims_i, dims_j, dims_k};

    std::shared_ptr<UniformTopology> utopo
      = std::make_shared<UniformTopology>(spacing, origin, dims);

    DataSet dataset(utopo);

    if(n_dataset.has_path("state/domain_id"))
    {
      dataset.domain_id(n_dataset["state/domain_id"].to_int32());
    }

    uniform_low_order_fields(n_dataset, dataset);
    return dataset;
  }



  template <typename T>
  T extract(const conduit::Node & node, size_t index)
  {
    return static_cast<const T *>(node.value())[index];
  }


  void uniform_low_order_fields(const conduit::Node &n_dataset, DataSet &dataset)
  {
    // we are assuming that this is uniform
    if(n_dataset.has_child("fields"))
    {
      // add all of the fields:
      NodeConstIterator itr = n_dataset["fields"].children();
      while(itr.has_next())
      {
        const Node &n_field = itr.next();
        std::string field_name = itr.name();

        const int num_windows = n_field["values"].number_of_children();

        if (num_windows == 0)
        {
          std::cout << "Expected the values of field "
                    << field_name
                    << " to be formatted as one or more windows.\n";
        }
        else
        {
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

          // Print an alert if the types of these arrays are not int64.
          {
            std::stringstream ss;
            bool oops = false;
            bool complain = false;
            ss << "Not int64: ";
            if ((oops = !n_values.child(0)["shape"].dtype().is_int64(),
                  (complain |= oops, oops)))
              ss << "shape ";
            if ((oops = !n_values.child(0)["strides"].dtype().is_int64(),
                (complain |= oops, oops)))
              ss << "strides ";
            if ((oops = !n_values.child(0)["origin"].dtype().is_int64(),
                (complain |= oops, oops)))
              ss << "origin ";
            ss << "\n";
            if (complain)
              std::cerr << ss.str();
          }
          using shape_t = conduit::int64;

          // Assumes that the set of windows occupies
          // a contiguous multi-D rectangle in index space.

          int num_axes = n_values.child(0)["shape"].dtype().number_of_elements();
          std::vector<size_t> total_origin(num_axes, std::numeric_limits<size_t>::max());
          std::vector<size_t> total_upper(num_axes, 0);
          unsigned long long total_size = 0;

          // Calculate sizes.
          for (int window = 0; window < num_windows; ++window)
          {
            const Node &n_window = n_values.child(window);
            size_t window_size = 1;
            for (int axis = 0; axis < num_axes; ++axis)
            {
              const size_t next_origin = extract<shape_t>(n_window["origin"], axis);
              const size_t next_shape = extract<shape_t>(n_window["shape"], axis);
              const size_t next_upper = next_origin + next_shape;
              if (total_origin[axis] > next_origin)
                total_origin[axis] = next_origin;
              if (total_upper[axis] < next_upper)
                total_upper[axis] = next_upper;
              window_size *= next_shape;
            }
            total_size += window_size;
          }

          std::vector<size_t> total_shape(num_axes, 0);
          unsigned long long size_from_shape = 1;
          for (int axis = 0; axis < num_axes; ++axis)
          {
            total_shape[axis] = total_upper[axis] - total_origin[axis];
            size_from_shape *= total_shape[axis];
          }

          // Verify contiguous block.
          if (size_from_shape != total_size)
          {
            std::stringstream size_ss;
            size_ss << "Extents [";
            for (int axis = 0; axis < num_axes; ++axis)
              size_ss << total_origin[axis] << ", ";
            size_ss << "] -- [";
            for (int axis = 0; axis < num_axes; ++axis)
              size_ss << total_upper[axis] << ", ";
            size_ss << "] do not match the actual size (" << total_size << ")";
            throw std::logic_error(size_ss.str());
          }
          else
          {
            std::cout << "Field " << field_name
                      << ": Sizes match (" << total_size << ").\n";
          }

          // "group" axis --> components
          // other axes --> items
          size_t num_groups = 1;
          size_t num_items = 1;
          int group_axis = -1;
          int zone_axis = -1;
          for (int axis = 0; axis < num_axes; ++axis)
          {
            const std::string label =
                n_values.child(0)["shape_labels"][axis].as_string();
            std::cout << "Label: '"<< label << "'\n";

            if (label == "zone")
              zone_axis = axis;

            if (label == "group")
            {
              group_axis = axis;
              num_groups = total_shape[axis];
            }
            else
            {
              num_items *= total_shape[axis];
            }
          }

          fprintf(stdout, "zone_axis==%d, group_axis==%d, "
                          "num_items==%lu, num_groups==%lu\n",
                          zone_axis, group_axis,
                          num_items, num_groups);

          // Allocate dray::Array with num_items items and num_groups components.
          Array<Float> all_windows;
          all_windows.resize(num_items, num_groups);

          // Zones should vary slowest.
          std::vector<int> axis_ordering(num_axes, 0);
          int true_axis = 0;
          for (int axis = 0; axis < num_axes; ++axis)
            if (axis != zone_axis)
              axis_ordering[true_axis++] = axis;
          if (zone_axis != -1)
            axis_ordering[num_axes-1] = zone_axis;

          // Respect axis ordering,
          // but map group axis to dray::Array components.
          ArrayFiller filler(num_axes, axis_ordering, total_origin, total_shape, group_axis);

          // Collect all windows in the field.
          for (int window = 0; window < num_windows; ++window)
          {
            const Node &n_window = n_values.child(window);
            if(!is_floating_point(n_window["data"]))
            {
              std::cout<<"non-floating point field '"<<field_name<<"'\n";
            }

            filler.fill_window<shape_t>(all_windows,
                                        n_window["data"],
                                        n_window["shape"],
                                        n_window["strides"],
                                        n_window["origin"]);
          }

          // Field association.
          std::string assoc_str = n_field["association"].as_string();
          LowOrderField::Assoc assoc;
          if(assoc_str == "vertex")
          {
            assoc = LowOrderField::Assoc::Vertex;
          }
          else
          {
            assoc = LowOrderField::Assoc::Element;
          }

          // Add dray field to dataset
          std::shared_ptr<LowOrderField> field
            = std::make_shared<LowOrderField>(all_windows, assoc);
          field->name(field_name);
          dataset.add_field(field);
        }

      } //while
    } // if has fields
  }

  bool is_floating_point(const conduit::Node &values)
  {
    return values.dtype().is_float32()
           || values.dtype().is_float64();
  }


  ArrayFiller::ArrayFiller(int num_axes,
                           const std::vector<int> &axis_ordering,
                           const std::vector<size_t> &total_origin,
                           const std::vector<size_t> &total_shape,
                           int component_axis)
    :
      m_total_origin(total_origin)
  {
    m_use_components = (component_axis != -1);
    m_component_axis = component_axis;

    m_dest_strides.resize(num_axes, 0);

    // Destination strides for item index.
    // Component axis maps to dray::Array component,
    // so does not contribute to item index.
    size_t stride = 1;
    for (int logical_axis = 0; logical_axis < num_axes; ++logical_axis)
    {
      const int true_axis = axis_ordering[logical_axis];
      if (true_axis != component_axis)
      {
        m_dest_strides[true_axis] = stride;
        stride *= total_shape[true_axis];
      }
    }
    if (component_axis != -1)
      m_dest_strides[component_axis] = 0;

    std::cout << "m_dest_strides: [";
    for (int axis = 0; axis < num_axes; ++axis)
      std::cout << m_dest_strides[axis] << ", ";
    std::cout << "]\n";
  }


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


  template <typename shape_t>
  void ArrayFiller::fill_window(Array<Float> all_windows,
                                const conduit::Node &n_window_data,
                                const conduit::Node &n_window_shape,
                                const conduit::Node &n_window_strides,
                                const conduit::Node &n_window_origin)
  {
    // Compute window offset into destination.
    size_t item_offset = 0;
    size_t component_offset = 0;

    const int num_axes = m_dest_strides.size();

    std::vector<shape_t> shape(num_axes);

    for (int axis = 0; axis < num_axes; ++axis)
    {
      shape[axis] = extract<shape_t>(n_window_shape, axis);
      const size_t window_origin = extract<shape_t>(n_window_origin, axis);
      const size_t window_offset = window_origin - m_total_origin[axis];
      if (m_use_components && axis == m_component_axis)
        component_offset = window_offset;
      else
        item_offset += m_dest_strides[axis] * window_offset;
    }

    // Copy value by value.
    if (n_window_data.dtype().is_float32())
    {
      copy_values<shape_t, float32>( all_windows,
                                     n_window_data,
                                     shape,
                                     n_window_strides,
                                     item_offset,
                                     component_offset);
    }
    else if (n_window_data.dtype().is_float64())
    {
      copy_values<shape_t, float64>( all_windows,
                                     n_window_data,
                                     shape,
                                     n_window_strides,
                                     item_offset,
                                     component_offset);
    }
  }

  template <typename shape_t, typename ValT>
  void ArrayFiller::copy_values(Array<Float> all_windows,
                                const conduit::Node &n_window_data,
                                const std::vector<shape_t> shape,
                                const conduit::Node &n_window_strides,
                                size_t item_offset,
                                size_t component_offset)
  {
    NonConstHostArray<Float> all_windows_data(all_windows);
    MultiDigit<shape_t> index(shape);
    const int num_axes = m_dest_strides.size();
    do
    {
      size_t src_idx = 0;
      size_t dest_item_idx = 0;
      for (int axis = 0; axis < num_axes; ++axis)
      {
        src_idx += index.digit(axis) * extract<shape_t>(n_window_strides, axis);
        dest_item_idx += index.digit(axis) * m_dest_strides[axis];
      }
      const size_t component = index.digit(m_component_axis);
      all_windows_data.get_item(item_offset + dest_item_idx,
                                component_offset + component)
          = extract<ValT>(n_window_data, src_idx);
      index.increment();
    }
    while (!index.is_zero());
  }


  Array<Float> fill_array(const conduit::Node &values)
  {
    Array<Float> res;

    if(!values.dtype().is_float32() &&
       !values.dtype().is_float64())
    {
      return res;
    }

    const int32 size = values.dtype().number_of_elements();
    res.resize(size);
    Float *res_ptr = res.get_host_ptr();

    if(values.dtype().is_float32())
    {
      const float32 *values_ptr = values.value();

      for(int32 i = 0; i < size; ++i)
      {
        res_ptr[i] = static_cast<Float>(values_ptr[i]);
      }
    }
    else
    {
      const float64 *values_ptr = values.value();

      for(int32 i = 0; i < size; ++i)
      {
        res_ptr[i] = static_cast<Float>(values_ptr[i]);
      }
    }

    return res;
  }




}
}


