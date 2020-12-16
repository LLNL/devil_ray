//
// Copyright (c) 2014-19, Lawrence Livermore National Security, LLC
// and Kripke project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//

#include <Kripke/ConduitInterface.h>

#include <Kripke.h>
#include <Kripke/Core/PartitionSpace.h>
#include <Kripke/ArchLayout.h>
#include <Kripke/Kernel.h>
#include <Kripke/ParallelComm.h>
#include <Kripke/Timing.h>
#include <Kripke/VarTypes.h>
#include <vector>
#include <sstream>
#include <stdio.h>

using namespace Kripke;
using namespace Kripke::Core;

/**
  Perform full parallel sweep algorithm on subset of subdomains.
*/

template<typename View>
void to_ndarray(conduit::Node &n_field, View &view)
{
  const int f_dims = view.layout.n_dims;

  int size = 1;
  for(int d = 0; d < f_dims; ++d)
  {
    size *= view.layout.sizes[d];
  }

  n_field["shape"].set(view.layout.sizes, f_dims);
  n_field["strides"].set(view.layout.strides, f_dims);
  n_field["data"].set_external(view.data, size);
}

void print_fields(const conduit::Node &dataset)
{
  const int num_spatial_doms = dataset.number_of_children();

  for(int i = 0; i < num_spatial_doms; ++i)
  {
    const conduit::Node &dom = dataset.child(i);
    const int num_fields = dom["fields"].number_of_children();
    for(int f = 0; f < num_fields; ++f)
    {
      const conduit::Node &field = dom["fields"].child(f);

      std::cout<<field.name()<<"\n";
      const int components = field["values"].number_of_children();
      if(components > 0)
      {
        for(int c = 0; c < components; ++c)
        {
          const conduit::Node &component = field["values"].child(c);
          std::cout<<" "<<component.name()<<"\n";
          std::cout<<"    shape     : "<<component["shape"].to_string()<<"\n";
          std::cout<<"    strides   : "<<component["strides"].to_string()<<"\n";
          std::cout<<"    origin    : "<<component["origin"].to_string()<<"\n";
          std::cout<<"    labels    : ";
          const int num_labels = component["shape_labels"].number_of_children();
          for(int l = 0; l < num_labels; ++l)
          {
            std::cout<<component["shape_labels"].child(l).to_string()<<" ";
          }
          std::cout<<"\n";
        } // components
      }
    } // fields
  } // domains
}

struct ToBPMesh
{
  template<typename AL>
  RAJA_INLINE
  void operator()(AL al,
                  Kripke::Core::DataStore &data_store,
                  Kripke::SdomId sdom_id,
                  conduit::Node &dom) const
  {
    auto sdom_al = getSdomAL(al, sdom_id);
    int local_imax = data_store.getVariable<Set>("Set/ZoneI").size(sdom_id);
    int local_jmax = data_store.getVariable<Set>("Set/ZoneJ").size(sdom_id);
    int local_kmax = data_store.getVariable<Set>("Set/ZoneK").size(sdom_id);

    int local_imin = data_store.getVariable<Set>("Set/ZoneI").lower(sdom_id);
    int local_jmin = data_store.getVariable<Set>("Set/ZoneJ").lower(sdom_id);
    int local_kmin = data_store.getVariable<Set>("Set/ZoneK").lower(sdom_id);


    auto dx = sdom_al.getView(data_store.getVariable<Field_ZoneI2Double>("dx"));
    auto dy = sdom_al.getView(data_store.getVariable<Field_ZoneJ2Double>("dy"));
    auto dz = sdom_al.getView(data_store.getVariable<Field_ZoneK2Double>("dz"));

    auto view_id = sdom_al.getView(data_store.getVariable<Field_Direction2Int>("quadrature/id"));
    auto view_jd = sdom_al.getView(data_store.getVariable<Field_Direction2Int>("quadrature/jd"));
    auto view_kd = sdom_al.getView(data_store.getVariable<Field_Direction2Int>("quadrature/kd"));

    // this never changes but I don't know how to ask for it
    double const x_min = -60.0;
    double const x_max = 60.0;

    double const y_min = -100.0;
    double const y_max = 100.0;

    double const z_min = -60.0;
    double const z_max = 60.0;

    // Assumption: all directions in this sdom have same mesh traversal
    Direction d0{0};
    int id = view_id(d0);
    int jd = view_jd(d0);
    int kd = view_kd(d0);

    ZoneI start_i((id>0) ? 0 : local_imax-1);
    ZoneJ start_j((jd>0) ? 0 : local_jmax-1);
    ZoneK start_k((kd>0) ? 0 : local_kmax-1);

    ZoneI end_i((id>0) ? local_imax : -1);
    ZoneJ end_j((jd>0) ? local_jmax : -1);
    ZoneK end_k((kd>0) ? local_kmax : -1);

    dom["topologies/topo/coordset"] = "coords";
    dom["topologies/topo/type"] = "uniform";

    dom["coordsets/coords/type"] = "uniform";
    dom["coordsets/coords/dims/i"] = local_imax + 1;
    dom["coordsets/coords/dims/j"] = local_jmax + 1;
    dom["coordsets/coords/dims/k"] = local_kmax + 1;
    dom["coordsets/coords/origin/x"] = x_min + dx(start_i) * local_imin;
    dom["coordsets/coords/origin/y"] = y_min + dy(start_j) * local_jmin;
    dom["coordsets/coords/origin/z"] = z_min + dz(start_k) * local_kmin;
    dom["coordsets/coords/spacing/dx"] = dx(start_i);
    dom["coordsets/coords/spacing/dy"] = dy(start_j);
    dom["coordsets/coords/spacing/dz"] = dz(start_k);
  }

};

struct ToBPMoments
{
  template<typename AL>
  RAJA_INLINE
  void operator()(AL al,
                  Kripke::Core::DataStore &data_store,
                  Kripke::SdomId sdom_id,
                  conduit::Node &dom) const
  {
    auto sdom_al = getSdomAL(al, sdom_id);

    int num_moments = data_store.getVariable<Set>("Set/Moment").size(sdom_id);
    int num_groups = data_store.getVariable<Set>("Set/Group").size(sdom_id);
    auto &set_group  = data_store.getVariable<Kripke::Core::Set>("Set/Group");
    auto &set_moment = data_store.getVariable<Kripke::Core::Set>("Set/Moment");

    auto field_phi = sdom_al.getView(data_store.getVariable<Field_Moments>("phi"));

    int phi_lower = set_moment.lower(sdom_id);
    int group_lower = set_group.lower(sdom_id);

    const int num_dims = set_moment.getNumDimensions();
    size_t dims[num_dims];
    for(int i = 0; i < num_dims; i++)
    {
      dims[i] = set_moment.dimSize(sdom_id,i);
    }

    std::stringstream ss;
    ss<<"phi_"<<phi_lower<<"_group_"<<group_lower;
    std::string phi_name = ss.str();

    conduit::Node &n_phi = dom["fields/phi"];
    n_phi["assocation"] = "element";
    n_phi["topology"] = "topo";
    conduit::Node &values = n_phi["values/"+phi_name];
    to_ndarray(values,field_phi);

    int origin[3] = {phi_lower, group_lower, 0};
    values["origin"].set(origin,3);
    values["shape_labels"].append() = "moment";
    values["shape_labels"].append() = "group";
    values["shape_labels"].append() = "zone";
  }

};

struct ToBPGroups
{
  template<typename AL>
  RAJA_INLINE
  void operator()(AL al,
                  Kripke::Core::DataStore &data_store,
                  Kripke::SdomId sdom_id,
                  conduit::Node &dom) const
  {
    auto sdom_al = getSdomAL(al, sdom_id);

    int num_directions = data_store.getVariable<Set>("Set/Direction").size(sdom_id);
    int num_groups = data_store.getVariable<Set>("Set/Group").size(sdom_id);
    auto &set_group  = data_store.getVariable<Kripke::Core::Set>("Set/Group");

    auto sigt = sdom_al.getView(data_store.getVariable<Kripke::Field_SigmaTZonal>("sigt_zonal"));

    int lower = set_group.lower(sdom_id);
    const int num_dims = set_group.getNumDimensions();
    size_t dims[num_dims];
    for(int i = 0; i < num_dims; i++)
    {
      dims[i] = set_group.dimSize(sdom_id,i);
    }

    std::stringstream ss;
    ss<<"group_"<<lower;
    std::string group_name = ss.str();

    conduit::Node &n_sigt = dom["fields/sigt"];
    n_sigt["assocation"] = "element";
    n_sigt["topology"] = "topo";
    conduit::Node &values = n_sigt["values/"+group_name];
    to_ndarray(values,sigt);

    int origin[2] = {lower, 0};
    values["origin"].set(origin,2);
    values["shape_labels"].append() = "group";
    values["shape_labels"].append() = "zone";
  }

};

void SpatialSdoms(Kripke::Core::DataStore &data_store,
                  conduit::Node &dataset,
                  std::map<SdomId,int> &sdom_to_dom)
{
  PartitionSpace &pspace = data_store.getVariable<PartitionSpace>("pspace");
  size_t num_space = pspace.getNumSubdomains(Core::SPACE::SPACE_R);
  std::vector<SdomId> subdomain_list;
  for(size_t i = 0; i < num_space; ++i)
  {
    subdomain_list.push_back(pspace.spaceToSubdomain(Core::SPACE::SPACE_R,i));
  }

  ArchLayoutV al_v = data_store.getVariable<ArchLayout>("al").al_v;

  int dom_counter = 0;
  for(auto &sdom_id : subdomain_list)
  {
    conduit::Node &dom = dataset.append();
    std::cout<<"Domain "<<dom_counter<<"\n";
    sdom_to_dom[sdom_id] = dom_counter;
    dom_counter++;
    Kripke::dispatch(al_v, ToBPMesh{}, data_store, sdom_id, dom);
  }
}

void GatherGroups(Kripke::Core::DataStore &data_store,
                  conduit::Node &dataset,
                  std::map<SdomId,int> &sdom_to_dom)
{
  PartitionSpace &pspace = data_store.getVariable<PartitionSpace>("pspace");
  size_t num_space = pspace.getNumSubdomains(Core::SPACE::SPACE_P);
  std::vector<SdomId> subdomain_list;
  for(size_t i = 0; i < num_space; ++i)
  {
    subdomain_list.push_back(pspace.spaceToSubdomain(Core::SPACE::SPACE_P,i));
  }

  ArchLayoutV al_v = data_store.getVariable<ArchLayout>("al").al_v;

  for(auto &sdom_id : subdomain_list)
  {
    int space_dom_idx = pspace.subdomainToSpace(Core::SPACE::SPACE_P, sdom_id);
    SdomId space_dom = pspace.spaceToSubdomain(Core::SPACE::SPACE_R, space_dom_idx);
    if(sdom_to_dom.find(space_dom) == sdom_to_dom.end())
    {
      std::cout<<"Can't find spatial MESH\n";
      continue;
    }
    int n_dom_id = sdom_to_dom[sdom_id];
    conduit::Node &dom = dataset.child(n_dom_id);;
    Kripke::dispatch(al_v, ToBPGroups{}, data_store, sdom_id, dom);
  }

}

void GatherMoments(Kripke::Core::DataStore &data_store,
                   conduit::Node &dataset,
                   std::map<SdomId,int> &sdom_to_dom)
{
  PartitionSpace &pspace = data_store.getVariable<PartitionSpace>("pspace");
  size_t num_space = pspace.getNumSubdomains(Core::SPACE::SPACE_PR);
  std::vector<SdomId> subdomain_list;
  for(size_t i = 0; i < num_space; ++i)
  {
    subdomain_list.push_back(pspace.spaceToSubdomain(Core::SPACE::SPACE_PR,i));
  }

  ArchLayoutV al_v = data_store.getVariable<ArchLayout>("al").al_v;

  for(auto &sdom_id : subdomain_list)
  {
    int space_dom_idx = pspace.subdomainToSpace(Core::SPACE::SPACE_PR, sdom_id);
    SdomId space_dom = pspace.spaceToSubdomain(Core::SPACE::SPACE_R, space_dom_idx);
    if(sdom_to_dom.find(space_dom) == sdom_to_dom.end())
    {
      std::cout<<"Can't find spatial MESH\n";
      continue;
    }
    int n_dom_id = sdom_to_dom[sdom_id];
    conduit::Node &dom = dataset.child(n_dom_id);;
    Kripke::dispatch(al_v, ToBPMoments{}, data_store, sdom_id, dom);
  }

}

void Kripke::ToBlueprint(Kripke::Core::DataStore &data_store,
                         conduit::Node &dataset)
{
  dataset.reset();

  std::map<SdomId,int> sdom_to_dom;
  // get the spatial mesh and create a mapping
  SpatialSdoms(data_store, dataset, sdom_to_dom);
  GatherGroups(data_store, dataset, sdom_to_dom);
  GatherMoments(data_store, dataset, sdom_to_dom);
  print_fields(dataset);
}

bool is_weird_for_vis(const conduit::Node &field)
{
  bool weird = false;
  int components = field["values"].number_of_children();
  if(components > 0)
  {
    if(field["values"].child(0).has_path("shape"))
    {
      weird = true;
    }
  }

  return weird;
}

void flatten_field(const conduit::Node &in_field, conduit::Node &outfield)
{
  //origins window/xyz
  std::vector<std::vector<int>> origins;
  //shapes window/xyz
  std::vector<std::vector<int>> shapes;
}

void Kripke::VisDump(Kripke::Core::DataStore &data_store)
{
  conduit::Node dataset;
  ToBlueprint(data_store, dataset);

}
