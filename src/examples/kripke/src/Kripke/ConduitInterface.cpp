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
#include <stdio.h>

using namespace Kripke;
using namespace Kripke::Core;

/**
  Perform full parallel sweep algorithm on subset of subdomains.
*/


struct ToBP
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

    int num_directions = data_store.getVariable<Set>("Set/Direction").size(sdom_id);
    int num_groups = data_store.getVariable<Set>("Set/Group").size(sdom_id);

    // default == 96 quaderature points
    // directions = 12
    // 96 / 12 = 8
    std::cout<<"Dirs "<<num_directions<<"\n";
    std::cout<<"Groups "<<num_groups<<"\n";

    auto dx = sdom_al.getView(data_store.getVariable<Field_ZoneI2Double>("dx"));
    auto dy = sdom_al.getView(data_store.getVariable<Field_ZoneJ2Double>("dy"));
    auto dz = sdom_al.getView(data_store.getVariable<Field_ZoneK2Double>("dz"));

    auto view_id = sdom_al.getView(data_store.getVariable<Field_Direction2Int>("quadrature/id"));
    auto view_jd = sdom_al.getView(data_store.getVariable<Field_Direction2Int>("quadrature/jd"));
    auto view_kd = sdom_al.getView(data_store.getVariable<Field_Direction2Int>("quadrature/kd"));

    auto sigt = sdom_al.getView(data_store.getVariable<Kripke::Field_SigmaTZonal>("sigt_zonal"));
    auto field_phi = sdom_al.getView(data_store.getVariable<Field_Moments>("phi"));
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

    std::cout<<"local_imax "<<local_imax<<"\n";
    std::cout<<"local_jmax "<<local_jmax<<"\n";
    std::cout<<"local_kmax "<<local_kmax<<"\n";

    std::cout<<"local_imin "<<local_imin<<"\n";
    std::cout<<"local_jmin "<<local_jmin<<"\n";
    std::cout<<"local_kmin "<<local_kmin<<"\n";

    std::cout<<"dx "<<dx(start_i)<<"\n";
    std::cout<<"dy "<<dy(start_j)<<"\n";
    std::cout<<"dz "<<dz(start_k)<<"\n";


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

    // do dirty things with templates
    constexpr int sigt_dims = Field_SigmaTZonal::DefaultLayoutType::Base::n_dims;
    std::cout<<"sigt_dims "<<sigt_dims<<"\n";

    int strides[sigt_dims];
    int shape[sigt_dims];
    int sigt_size = 1;
    for(int d = 0; d < sigt_dims; ++d)
    {
      //strides[d] = *Field_SigmaTZonal::DefaultLayoutType::Base::strides[d];
      // new raja = get_layout();
      strides[d] = sigt.layout.strides[d];
      shape[d] = sigt.layout.sizes[d];
      sigt_size *= shape[d];
    }

    conduit::Node &n_sigt = dom["fields/sigt"];
    n_sigt["assocation"] = "element";
    n_sigt["topology"] = "topo";
    n_sigt["shape"].set(sigt.layout.sizes, sigt_dims);
    n_sigt["strides"].set(sigt.layout.strides, sigt_dims);
    n_sigt["values"].set_external(sigt.data, sigt_size);

    //n_sigt.print();


  }

};

void Kripke::ToBlueprint(Kripke::Core::DataStore &data_store,
                         conduit::Node &dataset)
{
  dataset.reset();

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
    dom_counter++;
    Kripke::dispatch(al_v, ToBP{}, data_store, sdom_id, dom);
  }

}
