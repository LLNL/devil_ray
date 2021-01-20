// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <dray/io/blueprint_moments.hpp>
#include <dray/io/blueprint_uniform_topology.hpp>
#include <dray/io/array_mapping.hpp>

#include <dray/types.hpp>
#include <dray/error.hpp>
#include <dray/utils/data_logger.hpp>

#include <dray/GridFunction/low_order_field.hpp>

namespace dray
{
  namespace detail
  {
    // ================================================================
    // Helper Declarations
    // ================================================================
    static void uniform_low_order_fields(const conduit::Node &n_dataset, ArrayMapping & amap, DataSet &dataset, int32 &num_moments);
    // ================================================================


    // ================================================================
    // Definitions
    // ================================================================

    Collection import_into_uniform_moments(const conduit::Node &n_all_domains, ArrayMapping &amap, int32 &num_moments)
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
        DataSet dset = import_domain_into_uniform_moments(domain, amap, num_moments);
        collection.add_domain(dset);
      }

      return collection;
    }

    DataSet import_domain_into_uniform_moments(const conduit::Node &n_dataset, ArrayMapping &amap, int32 &num_moments)
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

      std::shared_ptr<UniformTopology> utopo
        = ::dray::detail::import_topology_into_uniform(topo, coords);

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
      using namespace conduit;

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


    void export_from_uniform_moments(const Collection &dray_collection, ArrayMapping &amap, conduit::Node &n_all_domains)
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
        export_from_uniform_moments(dray_domain, amap, n_domain);
      }
    }

    void export_from_uniform_moments(const DataSet &dray_dataset, ArrayMapping &amap, conduit::Node &n_dataset)
    {
      using namespace conduit;

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




  }//namespace detail
}//namespace dray
