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

template <int32 PhysDim>
GridFunction<PhysDim>
import_grid_function(const conduit::Node &n_gf,
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

  gf.from_node(n_gf);

  return gf;
}

void validate(const conduit::Node &node, std::vector<std::string> &info)
{

  // info[0] ==  topological dims
  // info[1] == tensor / simplex
  // info[2] == components
  // info[3] == order
  if(!node.has_path("type_name"))
  {
    DRAY_ERROR("Topology node has no type_name");
  }
  const std::string type_name = node["type_name"].as_string();

  std::cout<<"Type name "<<type_name<<"\n";
  info = detail::split(type_name, "_");;
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

  if(!node.has_path("grid_function"))
  {
    DRAY_ERROR("Topology missing grid function");
  }

  if(!node.has_path("order"))
  {
    DRAY_ERROR("Missing order");
  }
}

DataSet import_topology(const conduit::Node &n_topo)
{
  DataSet res;

  std::vector<std::string> info;
  validate(n_topo, info);

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
      GridFunction<3> gf = detail::import_grid_function<3>(n_gf, 3);
      using QuadMesh = MeshElem<2u, Tensor, General>;
      using QuadMesh_P1 = MeshElem<2u, Tensor, Linear>;
      using QuadMesh_P2 = MeshElem<2u, Tensor, Quadratic>;

      if(order == 1)
      {
        Mesh<QuadMesh_P1> mesh(gf, order);
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
      std::cout<<"Hex\n";
      GridFunction<3> gf = detail::import_grid_function<3>(n_gf, 3);
      using HexMesh = MeshElem<3u, Tensor, General>;
      using HexMesh_P1 = MeshElem<3u, Tensor, Linear>;
      using HexMesh_P2 = MeshElem<3u, Tensor, Quadratic>;

      if(order == 1)
      {
        Mesh<HexMesh_P1> mesh(gf, order);
        res = DataSet(std::make_shared<HexTopology_P1>(mesh));
      }
      else if(order == 2)
      {
        Mesh<HexMesh_P2> mesh(gf, order);
        res = DataSet(std::make_shared<HexTopology_P2>(mesh));
      }
      else
      {
        Mesh<HexMesh> mesh (gf, order);
        res = DataSet(std::make_shared<HexTopology>(mesh));
      }
    }
  }

  return res;
}

void import_field(const conduit::Node &n_field, DataSet &dataset)
{

  const std::string field_name = n_field.name();
  std::cout<<"Importing field "<<n_field.name()<<"\n";
  std::vector<std::string> info;
  validate(n_field, info);

  int32 order = n_field["order"].to_int32();

  const conduit::Node &n_gf = n_field["grid_function"];

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
      GridFunction<1> gf = detail::import_grid_function<1>(n_gf, 1);

      if(order == 1)
      {
        Field<QuadScalar_P1> field (gf, order, field_name);
        dataset.add_field(std::make_shared<Field<QuadScalar_P1>>(field));
      }
      else if(order == 2)
      {
        Field<QuadScalar_P2> field (gf, order, field_name);
        dataset.add_field(std::make_shared<Field<QuadScalar_P2>>(field));
      }
      else
      {
        Field<QuadScalar> field (gf, order, field_name);
        dataset.add_field(std::make_shared<Field<QuadScalar>>(field));
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
      std::cout<<"hex\n";
      GridFunction<1> gf = detail::import_grid_function<1>(n_gf, 1);

      if(order == 1)
      {
        Field<HexScalar_P1> field (gf, order, field_name);
        dataset.add_field(std::make_shared<Field<HexScalar_P1>>(field));
      }
      else if(order == 2)
      {
        Field<HexScalar_P2> field (gf, order, field_name);
        dataset.add_field(std::make_shared<Field<HexScalar_P2>>(field));
      }
      else
      {
        Field<HexScalar> field (gf, order, field_name);
        dataset.add_field(std::make_shared<Field<HexScalar>>(field));
      }
    }
  }

}

} // namspace detail

DataSet
to_dataset(const conduit::Node &n_dataset)
{
  n_dataset.print();
  if(!n_dataset.has_path("topology"))
  {
    DRAY_ERROR("Node has no topology");
  }
  const conduit::Node &n_topo = n_dataset["topology"];

  DataSet dataset = detail::import_topology(n_topo);
  if(n_dataset.has_path("fields"))
  {
    const int32 num_fields = n_dataset["fields"].number_of_children();
    for(int32 i = 0; i < num_fields; ++i)
    {
      detail::import_field(n_dataset["fields"].child(0), dataset);
    }
  }

  return dataset;
}

} // namespace dray
