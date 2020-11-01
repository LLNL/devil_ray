// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <dray/io/blueprint_low_order.hpp>
#include <dray/GridFunction/mesh.hpp>
#include <dray/derived_topology.hpp>
#include <dray/error.hpp>
#include "conduit_blueprint.hpp"

namespace dray
{

namespace detail
{

Array<Vec<Float,1>>
copy_conduit_scalar_array(const conduit::Node &n_vals)
{
  int num_vals = n_vals.dtype().number_of_elements();
  Array<Vec<Float,1>> values;
  values.resize(num_vals);

  Vec<Float,1> *values_ptr = values.get_host_ptr();

  if(n_vals.dtype().is_float32())
  {
    const float *n_values_ptr = n_vals.value();
    for(int32 i = 0; i < num_vals; ++i)
    {
      values_ptr[i][0] = n_values_ptr[i];
    }
  }
  else if(n_vals.dtype().is_float64())
  {
    const double *n_values_ptr = n_vals.value();
    for(int32 i = 0; i < num_vals; ++i)
    {
      values_ptr[i][0] = n_values_ptr[i];
    }
  }
  else
  {
    DRAY_ERROR("Unsupported copy type");
  }
  return values;
}

void
logical_index_2d(Vec<int32,3> &idx,
                 const int32 index,
                 const Vec<int32,3> &dims)
{
  idx[0] = index % dims[0];
  idx[1] = index / dims[0];
}

void
logical_index_3d(Vec<int32,3> &idx,
                 const int32 index,
                 const Vec<int32,3> &dims)
{
  idx[0] = index % dims[0];
  idx[1] = (index / dims[0]) % dims[1];
  idx[2] = index / (dims[0] * dims[1]);
}

} // namespace detail

DataSet
BlueprintLowOrder::import(const conduit::Node &n_dataset)
{
  DataSet dataset;

  conduit::Node info;
  if(!conduit::blueprint::verify("mesh",n_dataset, info))
  {
    DRAY_ERROR("Import failed to verify "<<info.to_yaml());
  }

  const int32 num_topos = n_dataset["topologies"].number_of_children();
  const conduit::Node &n_topo = n_dataset["topologies"].child(0);
  const std::string topo_name = n_dataset["topologies"].child_names()[0];

  const std::string coords_name = n_topo["coordset"].as_string();
  const string mesh_type = n_topo["type"].as_string();

  const conduit::Node &n_coords = n_dataset["coordsets/"+coords_name];

  Array<int32> conn;
  int32 n_elems = 0;
  bool is_2d = false;
  if(mesh_type == "uniform")
  {
    dataset = import_uniform(n_coords, conn, n_elems, is_2d);
  }
  else
  {
    DRAY_ERROR("not implemented "<<mesh_type);
  }


  const int32 num_fields = n_dataset["fields"].number_of_children();
  std::vector<std::string> field_names = n_dataset["fields"].child_names();
  for(int32 i = 0; i < num_fields; ++i)
  {
    const conduit::Node &n_field = n_dataset["fields"].child(i);
    std::string field_topo = n_field["topology"].as_string();

    if(field_topo != topo_name)
    {
      continue;
    }

    int32 components = n_field["values"].number_of_children();
    bool is_scalar = components == 0 || components == 1;

    if(!is_scalar)
    {
      std::cout<<"Skipping "<<field_names[i]<<" "<<components<<"\n";
    }

    std::string assoc = n_field["association"].as_string();
    if(assoc != "vertex")
    {
      std::cout<<"Skipping non-vertex fields for now\n";
    }

    const conduit::Node &n_vals = components == 0
      ? n_field["values"] : n_field["values"].child(0);

    Array<Vec<Float,1>> values = detail::copy_conduit_scalar_array(n_vals);


    int32 num_dofs = 1;

    // todo: this will depend on shape type
    if(assoc == "vertex")
    {
      num_dofs = 4;;
      if(!is_2d)
      {
        num_dofs = 8;
      }
    }

    GridFunction<1> gf;
    gf.m_ctrl_idx = conn;
    gf.m_values = values;
    gf.m_el_dofs = num_dofs;
    gf.m_size_el = n_elems;
    gf.m_size_ctrl = conn.size();
    int order = 1;

    if(is_2d)
    {
      std::shared_ptr<Field<QuadScalar_P1>> field
        = std::make_shared<Field<QuadScalar_P1>>(gf, order, field_names[i]);
      dataset.add_field(field);
    }
    else
    {
      std::shared_ptr<Field<HexScalar_P1>> field
        = std::make_shared<Field<HexScalar_P1>>(gf, order, field_names[i]);
      dataset.add_field(field);
    }

  }

  return dataset;
}

DataSet
BlueprintLowOrder::import_uniform(const conduit::Node &n_coords,
                                  Array<int32> &conn,
                                  int32 &n_elems,
                                  bool &is_2d)
{

  const std::string type = n_coords["type"].as_string();
  if(type != "uniform")
  {
    DRAY_ERROR("bad matt");
  }

  const conduit::Node &n_dims = n_coords["dims"];

  Vec<int32,3> dims;
  dims[0] = n_dims["i"].to_int();
  dims[1] = n_dims["j"].to_int();
  dims[2] = 1;

  is_2d = true;

  float64 origin_x = 0.0;
  float64 origin_y = 0.0;
  float64 origin_z = 0.0;

  float64 spacing_x = 1.0;
  float64 spacing_y = 1.0;
  float64 spacing_z = 1.0;

  if(n_coords.has_child("origin"))
  {
    const conduit::Node &n_origin = n_coords["origin"];

    if(n_origin.has_child("x"))
    {
      origin_x = n_origin["x"].to_float64();
    }

    if(n_origin.has_child("y"))
    {
      origin_y = n_origin["y"].to_float64();
    }

    if(n_origin.has_child("z"))
    {
      origin_z = n_origin["z"].to_float64();
    }
  }

  if(n_coords.has_path("spacing"))
  {
    const conduit::Node &n_spacing = n_coords["spacing"];

    if(n_spacing.has_path("dx"))
    {
      spacing_x = n_spacing["dx"].to_float64();
    }

    if(n_spacing.has_path("dy"))
    {
      spacing_y = n_spacing["dy"].to_float64();
    }

    if(n_spacing.has_path("dz"))
    {
      spacing_z = n_spacing["dz"].to_float64();
    }
  }

  Array<Vec<Float,3>> coords;
  const int32 n_verts = dims[0] * dims[1] * dims[2];
  coords.resize(n_verts);
  Vec<Float,3> *coords_ptr = coords.get_host_ptr();

  for(int32 i = 0; i < n_verts; ++i)
  {
    Vec<int32,3> idx;
    detail::logical_index_3d(idx, i, dims);

    Vec<Float,3> point;
    point[0] = origin_x + idx[0] * spacing_x;
    point[1] = origin_y + idx[1] * spacing_y;
    point[2] = origin_z + idx[2] * spacing_z;

    coords_ptr[i] = point;
  }

  n_elems = (dims[0] - 1) * (dims[1] - 1);
  if(!is_2d)
  {
    n_elems *= dims[2] - 1;
  }

  const int32 verts_per_elem = is_2d ? 4 : 8;

  conn.resize(n_verts * verts_per_elem);
  int32 *conn_ptr = conn.get_host_ptr();

  for(int32 i = 0; i < n_elems; ++i)
  {
    const int32 offset = i * verts_per_elem;
    Vec<int32,3> idx;

    if(is_2d)
    {
      detail::logical_index_2d(idx, i, dims);
      // this is the vtk version
      //conn_ptr[offset + 0] = idx[1] * dims[0] + idx[0];
      //conn_ptr[offset + 1] = conn_ptr[offset + 0] + 1;
      //conn_ptr[offset + 2] = conn_ptr[offset + 1] + dims[0];
      //conn_ptr[offset + 3] = conn_ptr[offset + 2] - 1;
      // this is the dray version (lexagraphical ordering x,y,z)
      conn_ptr[offset + 0] = idx[1] * dims[0] + idx[0];
      conn_ptr[offset + 1] = conn_ptr[offset + 0] + 1;
      conn_ptr[offset + 2] = conn_ptr[offset + 0] + dims[0];
      conn_ptr[offset + 3] = conn_ptr[offset + 2] + 1;
    }
    else
    {
      detail::logical_index_3d(idx, i, dims);
      // this is the vtk version
      //conn_ptr[offset + 0] = (idx[2] * dims[1] + idx[1]) * dims[0] + idx[0];
      //conn_ptr[offset + 1] = conn_ptr[offset + 0] + 1;
      //conn_ptr[offset + 2] = conn_ptr[offset + 1] + dims[1];
      //conn_ptr[offset + 3] = conn_ptr[offset + 2] - 1;
      //conn_ptr[offset + 4] = conn_ptr[offset + 0] + dims[0] * dims[2];
      //conn_ptr[offset + 5] = conn_ptr[offset + 4] + 1;
      //conn_ptr[offset + 6] = conn_ptr[offset + 5] + dims[1];
      //conn_ptr[offset + 7] = conn_ptr[offset + 6] - 1;
      // this is the dray version (lexagraphical ordering x,y,z)
      conn_ptr[offset + 0] = (idx[2] * dims[1] + idx[1]) * dims[0] + idx[0];
      conn_ptr[offset + 1] = conn_ptr[offset + 0] + 1;

      // advance in y
      conn_ptr[offset + 2] = conn_ptr[offset + 0] + dims[0];
      conn_ptr[offset + 3] = conn_ptr[offset + 2] + 1;

      // advance in z
      conn_ptr[offset + 4] = conn_ptr[offset + 0] + dims[0] * dims[2];
      conn_ptr[offset + 5] = conn_ptr[offset + 4] + 1;
      // advance in y
      conn_ptr[offset + 6] = conn_ptr[offset + 5] + dims[0];
      conn_ptr[offset + 7] = conn_ptr[offset + 6] + 1;
    }
  }

  GridFunction<3> gf;
  gf.m_ctrl_idx = conn;
  gf.m_values = coords;
  gf.m_el_dofs = verts_per_elem;
  gf.m_size_el = n_elems;
  gf.m_size_ctrl = conn.size();

  using HexMesh = MeshElem<3u, Tensor, Linear>;
  using QuadMesh = MeshElem<2u, Tensor, Linear>;
  int32 order = 1;

  DataSet res;
  if(is_2d)
  {
    Mesh<QuadMesh> mesh (gf, order);
    std::shared_ptr<QuadTopology_P1> topo = std::make_shared<QuadTopology_P1>(mesh);
    DataSet dataset(topo);
    res = dataset;
  }
  else
  {
    Mesh<HexMesh> mesh (gf, order);
    std::shared_ptr<HexTopology_P1> topo = std::make_shared<HexTopology_P1>(mesh);
    DataSet dataset(topo);
    res = dataset;
  }

  return res;
}

} // namespace dray
