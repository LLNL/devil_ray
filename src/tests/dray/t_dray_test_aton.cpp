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


namespace aton
{
namespace detail
{
  using namespace dray;

  // --------------------------------------------------------
  DataSet import_into_uniform(const conduit::Node &n_all_domains);
  DataSet import_domain_into_uniform(const conduit::Node &n_dataset);
  void uniform_low_order_fields(const conduit::Node &n_dataset, DataSet &dataset);
  Array<Float> fill_array(const conduit::Node &values);
  // --------------------------------------------------------
}
}


TEST(aton_dray, aton_import)
{
  conduit::Node data;
  conduit::relay::io::load("../../../debug/external_source.json", "json", data);

  dray::DataSet dray_dataset = aton::detail::import_into_uniform(data);

}



namespace aton
{
namespace detail
{
  using namespace dray;

  DataSet import_into_uniform(const conduit::Node &n_all_domains)
  {
    const int doms = n_all_domains.number_of_children();

    if (doms == 0)
    {
      DRAY_ERROR("empty domain list");
    }
    else if (doms > 1)
    {
      throw std::logic_error("aton import_into_uniform() domain overloading "
                             "(doms>1) not yet implemented.");
    }

    return import_domain_into_uniform(n_all_domains.child(0));
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
    uniform_low_order_fields(n_dataset, dataset);
    return dataset;
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
          if (complain)
            std::cout << ss.str();

          //TODO count the total shape and allocate a dray array
          //  use the labels to find "group", make this total size
          //  the ncomp in dray::array constructor.
          Array<Float> all_windows;

          // Collect all windows in the field.
          for (int window = 0; window < num_windows; ++window)
          {
            const Node &n_window = n_values.child(window);
            Array<Float> window_data = fill_array(n_window["data"]);
            if(window_data.size() == 0)
            {
              std::cout<<"non-floating point field '"<<field_name<<"'\n";
            }

            // TODO copy window_data to its place in the total array
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


