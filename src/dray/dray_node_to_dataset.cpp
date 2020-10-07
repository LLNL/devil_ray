// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <dray/dray_node_to_dataset.hpp>
#include <dray/error.hpp>
#include <dray/Element/elem_attr.hpp>
#include <dray/GridFunction/grid_function.hpp>
#include <dray/GridFunction/mesh.hpp>
#include <dray/derived_topology.hpp>

namespace dray
{

namespace detail
{

std::vector<std::string> split (std::string s, std::string delimiter)
{
  size_t pos_start = 0, pos_end, delim_len = delimiter.length();
  string token;
  std::vector<string> res;

  while ((pos_end = s.find (delimiter, pos_start)) != string::npos)
  {
    token = s.substr (pos_start, pos_end - pos_start);
    pos_start = pos_end + delim_len;
    res.push_back (token);
  }

  res.push_back (s.substr (pos_start));
  return res;
}

template <int32 PhysDim, int32 RefDim>
GridFunction<PhysDim>
import_grid_function(const conduit::Node &n_gf,
                     ElemType elem_type,
                     int compoments)
{
  GridFunction<PhysDim> gf;
  if(!n_gf.has_path("values"))
  {
    DRAY_ERROR("Grid function missing values");
  }
  if(!n_gf.has_path("conn"))
  {
    DRAY_ERROR("Grid function missing connectivity");
  }

  gf.m_el_dofs = n_gf["dofs_per_element"].to_int32();
  gf.m_size_el = n_gf["num_elements"].to_int32();

  Float *values_ptr = (Float *) n_gf["values"].data_ptr();
  int32 values_size = n_gf["values"].dtype().number_of_elements();
  Vec<Float, PhysDim> *vec_values_ptr = (Vec<Float,PhysDim>*)values_ptr;
  gf.m_values.set(vec_values_ptr, values_size / PhysDim);

  int32 ctrl_size = n_gf["conn"].dtype().number_of_elements();
  int32 *ctrl_ptr = (int32*) n_gf["conn"].data_ptr();
  gf.m_ctrl_idx.set(ctrl_ptr, ctrl_size);

  return gf;
}

DataSet import_topology(const conduit::Node &n_topo)
{
  DataSet res;
  if(!n_topo.has_path("type_name"))
  {
    DRAY_ERROR("Topology node has no type_name");
  }
  const std::string type_name = n_topo["type_name"].as_string();

  std::cout<<"Type name "<<type_name<<"\n";
  std::vector<std::string> info = detail::split(type_name, "_");;
  // info[0] ==  topological dims
  // info[1] == tensor / simplex
  // info[2] == components
  // info[3] == order
  for(int i = 0; i < info.size(); ++i)
  {
    std::cout<<info[i]<<"\n";
  }

  if(info[0] != "2D" && info[0] != "3D")
  {
    DRAY_ERROR("Unknown topological dim:'"<<info[0]<<"'");
  }

  if(info[1] != "Simplex" && info[1] != "Tensor")
  {
    DRAY_ERROR("Unknown element type :'"<<info[1]<<"'");
  }

  if(!n_topo.has_path("grid_function"))
  {
    DRAY_ERROR("Topology missing grid function");
  }

  if(!n_topo.has_path("order"))
  {
    DRAY_ERROR("Topology missing order");
  }

  int32 order = n_topo["order"].to_int32();

  const conduit::Node &n_gf = n_topo["grid_function"];

  if(info[0] == "2D")
  {
    if(info[1] == "Simplex")
    {
      // triangle
    }
    else
    {
      // quad
      std::cout<<"Quad\n";
      GridFunction<3> gf = detail::import_grid_function<3,2>(n_gf, Simplex, 3);
      using QuadMesh = MeshElem<2u, Tensor, General>;
      using QuadMesh_P1 = MeshElem<2u, Tensor, Linear>;
      using QuadMesh_P2 = MeshElem<2u, Tensor, Quadratic>;

      if(order == 1)
      {
        Mesh<QuadMesh_P1> mesh(gf, order);
        //res = DataSet(std::make_shared<QuadTopology_P1>(mesh.template to_fixed_order<1>()));
        res = DataSet(std::make_shared<QuadTopology_P1>(mesh));
      }
      else if(order == 2)
      {
        Mesh<QuadMesh_P2> mesh(gf, order);
        res = DataSet(std::make_shared<QuadTopology_P2>(mesh));
      }
      else
      {
        Mesh<QuadMesh> mesh (gf, order);
        res = DataSet(std::make_shared<QuadTopology>(mesh));
      }
    }
  }
  else if(info[0] == "3D")
  {
    if(info[1] == "Simplex")
    {
      // tet
    }
    else
    {
      // hex
    }
  }

  return res;
}

} // namspace detail

//DataSet
void to_dataset(const conduit::Node &n_dataset)
{
  n_dataset.print();
  if(!n_dataset.has_path("topology"))
  {
    DRAY_ERROR("Node has no topology");
  }
  const conduit::Node &n_topo = n_dataset["topology"];


}

} // namespace dray
