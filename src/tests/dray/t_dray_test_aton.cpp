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
  class ArrayMapping;

  Collection import_into_uniform(const conduit::Node &n_all_domains, ArrayMapping & amap, int32 &num_moments);
  DataSet import_domain_into_uniform(const conduit::Node &n_dataset, ArrayMapping & amap, int32 &num_moments);
  void uniform_low_order_fields(const conduit::Node &n_dataset, ArrayMapping & amap, DataSet &dataset, int32 &num_moments);

  void export_from_uniform(const Collection &dray_collection, ArrayMapping & amap, conduit::Node &n_all_domains);
  void export_from_uniform(const DataSet &dray_dataset, ArrayMapping & amap, conduit::Node &n_dataset);

  bool is_floating_point(const conduit::Node &values);



  template <typename T>
  T extract(const conduit::Node & node, size_t index)
  {
    return static_cast<const T *>(node.value())[index];
  }

  template <typename T>
  T & extract_ref(conduit::Node & node, size_t index)
  {
    return static_cast<T *>(node.value())[index];
  }




  struct WindowBlockMeta
  {
    WindowBlockMeta() = default;

    // registration subtask
    template <typename ShapeT>
    WindowBlockMeta(const std::string &field_name,
                    const conduit::Node &n_field,
                    ShapeT shape_t_tag);

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

  struct WindowMeta
  {
    WindowMeta() = default;

    // registration subtask
    template <typename ShapeT>
    WindowMeta(const std::string &field_name,
               const conduit::Node &n_window,
               const WindowBlockMeta &block_meta,
               ShapeT shape_t_tag);

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

  // WindowBlock
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


  //
  // ArrayMapping
  //
  class ArrayMapping
  {
    public:
      // ------------------------------------------------------------
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
      // ------------------------------------------------------------

    protected:
      using DomainIndexT = int32;
      using FieldNameT = std::string;

      int32 m_active_domain = 0;

      std::map<DomainIndexT,
               std::map<FieldNameT,
                        WindowBlock>> m_domains;
  };

  // --------------------------------------------------------
}
}


TEST(aton_dray, aton_import_and_integrate)
{
  conduit::Node data;
  conduit::relay::io::load("../../../debug/external_source.json", "json", data);

  dray::int32 num_moments;
  aton::detail::ArrayMapping amap;
  dray::Collection dray_collection = aton::detail::import_into_uniform(data, amap, num_moments);

  dray::FirstScatter first_scatter;
  first_scatter.emission_field("phi");
  first_scatter.total_cross_section_field("sigt");
  first_scatter.legendre_order(sqrt(num_moments) - 1);
  first_scatter.uniform_isotropic_scattering(0.05f);  // TODO don't assume uniform scattering

  first_scatter.overwrite_first_scatter_field("first_scatter");
  first_scatter.execute(dray_collection);

  aton::detail::export_from_uniform(dray_collection, amap, data);

  conduit::relay::io::save(data, "first_scatter_source.json");
}



namespace aton
{
namespace detail
{
  using namespace dray;


  Collection import_into_uniform(const conduit::Node &n_all_domains, ArrayMapping &amap, int32 &num_moments)
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
      amap.active_domain(domain_id);
      DataSet dset = import_domain_into_uniform(domain, amap, num_moments);
      collection.add_domain(dset);
    }

    return collection;
  }

  DataSet import_domain_into_uniform(const conduit::Node &n_dataset, ArrayMapping &amap, int32 &num_moments)
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

    uniform_low_order_fields(n_dataset, amap, dataset, num_moments);
    return dataset;
  }



  void uniform_low_order_fields(const conduit::Node &n_dataset, ArrayMapping &amap, DataSet &dataset, int32 &num_moments)
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

        const WindowBlock & window_block =
            amap.register_uniform_low_order_field(field_name, n_field);

        Array<Float> all_windows;
        window_block.resize_dray_array(all_windows);
        window_block.copy_conduit_2_dray(n_field, all_windows);

        //TODO get legendre_order directly
        num_moments = window_block.num_moments();

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

      } //while
    } // if has fields
  }


  void export_from_uniform(const Collection &dray_collection, ArrayMapping &amap, conduit::Node &n_all_domains)
  {
    const int doms = n_all_domains.number_of_children();

    if (doms == 0)
    {
      DRAY_ERROR("empty domain list");
    }

    for(int i = 0; i < doms; ++i)
    {
      conduit::Node &n_domain = n_all_domains.child(i);
      int domain_id = 0;
      if(n_domain.has_path("state/domain_id"))
        domain_id = n_domain["state/domain_id"].to_int32();
      DRAY_INFO("Exporting domain "<<domain_id);
      amap.active_domain(domain_id);
      const DataSet &dray_domain = dray_collection.domain(domain_id);
      export_from_uniform(dray_domain, amap, n_domain);
    }
  }

  void export_from_uniform(const DataSet &dray_dataset, ArrayMapping &amap, conduit::Node &n_dataset)
  {
    // This is pull, not push.
    // Copy only, and all of, the fields in the conduit node.

    // we are assuming that this is uniform
    if(n_dataset.has_child("fields"))
    {
      // all of the fields:
      NodeIterator itr = n_dataset["fields"].children();
      while(itr.has_next())
      {
        Node &n_field = itr.next();
        std::string field_name = itr.name();

        const WindowBlock & window_block =
            amap.retrieve_uniform_low_order_field(field_name);

        const bool has_field = dray_dataset.has_field(field_name);
        if (has_field)
        {
          // Get field from dray dataset
          FieldBase *field_base = DataSet(dray_dataset).field(field_name);
          LowOrderField *field_lo =
              dynamic_cast<LowOrderField *>(field_base);
          const Array<Float> all_windows = field_lo->values();

          // Put back into conduit.
          window_block.copy_dray_2_conduit(all_windows, n_field);
        }

      } //while
    } // if has fields
  }




  bool is_floating_point(const conduit::Node &values)
  {
    return values.dtype().is_float32()
           || values.dtype().is_float64();
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
    WindowBlock &window_block = m_domains[m_active_domain][field_name];
    window_block = WindowBlock(field_name, n_field);

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

    return m_domains.find(m_active_domain)->second.find(field_name)->second;
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

    m_meta = WindowBlockMeta(field_name, n_field, shape_t{});

    // Add window-specific strides.
    for (int window = 0; window < num_windows; ++window)
      m_windows.emplace_back(field_name,
                             n_values.child(window),
                             m_meta,
                             shape_t{});
  }

  //
  // WindowBlockMeta() constructor
  //
  template <typename ShapeT>
  WindowBlockMeta::WindowBlockMeta(const std::string &field_name,
                                   const conduit::Node &n_field,
                                   ShapeT shape_t_tag)
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
        const size_t next_origin = extract<ShapeT>(n_window["origin"], axis);
        const size_t next_shape = extract<ShapeT>(n_window["shape"], axis);
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
  // WindowMeta() constructor
  //
  template <typename ShapeT>
  WindowMeta::WindowMeta(const std::string &field_name,
                         const conduit::Node &n_window,
                         const WindowBlockMeta &block,
                         ShapeT shape_t_tag)
  {
    m_item_offset = 0;
    m_component_offset = 0;

    const int num_axes = block.m_num_axes;
    m_conduit_shape.resize(num_axes);
    m_conduit_strides.resize(num_axes);

    for (int axis = 0; axis < num_axes; ++axis)
    {
      m_conduit_shape[axis] = extract<ShapeT>(n_window["shape"], axis);
      m_conduit_strides[axis] = extract<ShapeT>(n_window["strides"], axis);
      const size_t window_origin = extract<ShapeT>(n_window["origin"], axis);
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

      extract_ref<ValT>(n_window_data, conduit_idx)
        = all_windows_data.get_item(m_item_offset + dray_item_idx,
                                    m_component_offset + component);

      index.increment();
    }
    while (!index.is_zero());
  }



}
}


