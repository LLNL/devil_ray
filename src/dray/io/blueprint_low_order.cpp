// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <dray/io/blueprint_low_order.hpp>
#include <dray/GridFunction/mesh.hpp>
#include <dray/derived_topology.hpp>
#include <dray/error.hpp>
#include <dray/array_utils.hpp>
#include "conduit_blueprint.hpp"

namespace dray
{

namespace detail
{

// vtk to lexagraphical ordering (zyx)
constexpr int32 hex_conn_map[8] = {0, 1, 4, 5, 3, 2, 7, 6 };
constexpr int32 quad_conn_map[4] = {0, 1, 3, 2};
constexpr int32 tet_conn_map[4] = {0, 1, 2, 3};
constexpr int32 tri_conn_map[3] = {0, 1, 2};

int32 dofs_per_elem(const std::string shape)
{
  int32 dofs = 0;
  if(shape == "tri")
  {
    dofs = 3;
  }
  else if(shape == "tet" || shape == "quad")
  {
    dofs = 4;
  }
  else if(shape == "hex")
  {
    dofs = 8;
  }
  return dofs;
}

template<typename T>
Array<int32>
convert_conn(const conduit::Node &n_conn,
             const std::string shape,
             int32 &num_elems)
{
  const int conn_size = n_conn.dtype().number_of_elements();
  Array<int32> conn;
  conn.resize(conn_size);
  int32 *conn_ptr = conn.get_host_ptr();

  const int num_dofs = dofs_per_elem(shape);
  num_elems = conn_size / num_dofs;

  conduit::DataArray<int32> conn_array = n_conn.value();

  const int32 *map = shape == "hex" ? hex_conn_map :
                     shape == "quad" ? quad_conn_map :
                     shape == "tri" ? tri_conn_map : tet_conn_map;
  for(int32 i = 0; i < num_elems; ++i)
  {
    const int32 offset = i * num_dofs;
    for(int32 dof = 0; dof < num_dofs; ++dof)
    {
      conn_ptr[offset + dof] = conn_array[offset + map[dof]];
    }
  }

  return conn;
}

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

Array<Vec<Float,3>>
import_explicit_coords(const conduit::Node &n_coords)
{
    int32 nverts = n_coords["values/x"].dtype().number_of_elements();

    Array<Vec<Float,3>> coords;
    coords.resize(nverts);
    Vec<Float,3> *coords_ptr = coords.get_host_ptr();

    int32 ndims = 2;
    if(n_coords["values"].has_path("z"))
    {
      ndims = 3;
    }

    bool is_float = n_coords["values/x"].dtype().is_float32();

    if(is_float)
    {
      conduit::float32_array x_array = n_coords["values/x"].value();
      conduit::float32_array y_array = n_coords["values/y"].value();
      conduit::float32_array z_array;

      if(ndims == 3)
      {
        z_array = n_coords["values/z"].value();
      }

      for(int32 i = 0; i < nverts; ++i)
      {
        Vec<Float,3> point;
        point[0] = x_array[i];
        point[1] = y_array[i];
        point[2] = 0.f;
        if(ndims == 3)
        {
          point[2] = z_array[i];
        }
        coords_ptr[i] = point;
      }
    }
    else
    {
      conduit::float64_array x_array = n_coords["values/x"].value();
      conduit::float64_array y_array = n_coords["values/y"].value();
      conduit::float64_array z_array;

      if(ndims == 3)
      {
        z_array = n_coords["values/z"].value();
      }

      for(int32 i = 0; i < nverts; ++i)
      {
        Vec<Float,3> point;
        point[0] = x_array[i];
        point[1] = y_array[i];
        point[2] = 0.f;
        if(ndims == 3)
        {
          point[2] = z_array[i];
        }

        coords_ptr[i] = point;
      }
    }

    return coords;
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
  const std::string mesh_type = n_topo["type"].as_string();

  const conduit::Node &n_coords = n_dataset["coordsets/"+coords_name];

  Array<int32> conn;
  int32 n_elems = 0;
  std::string shape;
  if(mesh_type == "uniform")
  {
    dataset = import_uniform_to_explicit(n_coords, conn, n_elems, shape);
  }
  else if(mesh_type == "unstructured")
  {
    dataset = import_explicit_to_explicit(n_coords,
                              n_topo,
                              conn,
                              n_elems,
                              shape);
  }
  else
  {
    DRAY_ERROR("not implemented "<<mesh_type);
  }


  const int32 num_fields = n_dataset["fields"].number_of_children();
  std::vector<std::string> field_names = n_dataset["fields"].child_names();

  Array<int32> element_conn;

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

    int order = 1;
    if(assoc != "vertex" )
    {
      order = 0;
      if(element_conn.size() == 0)
      {
        element_conn = array_counting(n_elems, 0, 1);
      }
    }

    const conduit::Node &n_vals = components == 0
      ? n_field["values"] : n_field["values"].child(0);

    Array<Vec<Float,1>> values = detail::copy_conduit_scalar_array(n_vals);


    int32 num_dofs = 1;

    // todo: this will depend on shape type
    if(assoc == "vertex")
    {
      num_dofs = detail::dofs_per_elem(shape);
    }

    GridFunction<1> gf;
    gf.m_ctrl_idx = assoc == "vertex" ? conn : element_conn;
    gf.m_values = values;
    gf.m_el_dofs = num_dofs;
    gf.m_size_el = n_elems;
    gf.m_size_ctrl = conn.size();

    if(shape == "quad")
    {
      if(assoc == "vertex")
      {
        std::shared_ptr<Field<QuadScalar_P1>> field
          = std::make_shared<Field<QuadScalar_P1>>(gf, order, field_names[i]);
        dataset.add_field(field);
      }
      else
      {
        std::shared_ptr<Field<QuadScalar_P0>> field
          = std::make_shared<Field<QuadScalar_P0>>(gf, order, field_names[i]);
        dataset.add_field(field);
      }
    }
    else if(shape == "hex")
    {
      if(assoc == "vertex")
      {
        std::shared_ptr<Field<HexScalar_P1>> field
          = std::make_shared<Field<HexScalar_P1>>(gf, order, field_names[i]);
        dataset.add_field(field);
      }
      else
      {
        std::shared_ptr<Field<HexScalar_P0>> field
          = std::make_shared<Field<HexScalar_P0>>(gf, order, field_names[i]);
        dataset.add_field(field);
      }
    }
    else if(shape == "tri")
    {
      if(assoc == "vertex")
      {
        std::shared_ptr<Field<TriScalar_P1>> field
          = std::make_shared<Field<TriScalar_P1>>(gf, order, field_names[i]);
        dataset.add_field(field);
      }
      else
      {
        std::shared_ptr<Field<TriScalar_P0>> field
          = std::make_shared<Field<TriScalar_P0>>(gf, order, field_names[i]);
        dataset.add_field(field);
      }
    }
    else if(shape == "tet")
    {
      if(assoc == "vertex")
      {
        std::shared_ptr<Field<TetScalar_P1>> field
          = std::make_shared<Field<TetScalar_P1>>(gf, order, field_names[i]);
        dataset.add_field(field);
      }
      else
      {
        std::shared_ptr<Field<TetScalar_P0>> field
          = std::make_shared<Field<TetScalar_P0>>(gf, order, field_names[i]);
        dataset.add_field(field);
      }
    }
  }
  return dataset;
}

DataSet
BlueprintLowOrder::import_explicit_to_explicit(const conduit::Node &n_coords,
                                   const conduit::Node &n_topo,
                                  Array<int32> &conn,
                                  int32 &n_elems,
                                  std::string &shape)
{
  const std::string type = n_coords["type"].as_string();
  if(type != "explicit")
  {
    DRAY_ERROR("bad matt");
  }

  Array<Vec<Float,3>> coords = detail::import_explicit_coords(n_coords);

  const conduit::Node &n_topo_eles = n_topo["elements"];
  std::string ele_shape = n_topo_eles["shape"].as_string();
  shape = ele_shape;
  bool supported_shape = false;

  if(ele_shape == "hex" || ele_shape == "tet" ||
     ele_shape ==  "quad" || ele_shape == "tri")
  {
    supported_shape = true;
  }

  if(!supported_shape)
  {
    DRAY_ERROR("Shape '"<<ele_shape<<"' not currently supported");
  }

  const conduit::Node &n_topo_conn = n_topo_eles["connectivity"];
  n_elems = 0;
  if(n_topo_conn.dtype().is_int32())
  {
    conn = detail::convert_conn<int32>(n_topo_conn, ele_shape, n_elems);
  }
  else if(n_topo_conn.dtype().is_int64())
  {
    conn = detail::convert_conn<int64>(n_topo_conn, ele_shape, n_elems);
  }
  else
  {
    DRAY_ERROR("Unsupported conn data type");
  }

  int32 verts_per_elem = detail::dofs_per_elem(ele_shape);

  GridFunction<3> gf;
  gf.m_ctrl_idx = conn;
  gf.m_values = coords;
  gf.m_el_dofs = verts_per_elem;
  gf.m_size_el = n_elems;
  gf.m_size_ctrl = conn.size();

  using HexMesh = MeshElem<3u, Tensor, Linear>;
  using QuadMesh = MeshElem<2u, Tensor, Linear>;
  using TetMesh = MeshElem<3u, Simplex, Linear>;
  using TriMesh = MeshElem<2u, Simplex, Linear>;
  int32 order = 1;

  DataSet res;
  if(ele_shape == "tri")
  {
    Mesh<TriMesh> mesh (gf, order);
    std::shared_ptr<TriTopology_P1> topo = std::make_shared<TriTopology_P1>(mesh);
    DataSet dataset(topo);
    res = dataset;
  }
  else if(ele_shape == "tet")
  {
    Mesh<TetMesh> mesh (gf, order);
    std::shared_ptr<TetTopology_P1> topo = std::make_shared<TetTopology_P1>(mesh);
    DataSet dataset(topo);
    res = dataset;
  }
  else if(ele_shape == "quad")
  {
    Mesh<QuadMesh> mesh (gf, order);
    std::shared_ptr<QuadTopology_P1> topo = std::make_shared<QuadTopology_P1>(mesh);
    DataSet dataset(topo);
    res = dataset;
  }
  else if(ele_shape == "hex")
  {
    Mesh<HexMesh> mesh (gf, order);
    std::shared_ptr<HexTopology_P1> topo = std::make_shared<HexTopology_P1>(mesh);
    DataSet dataset(topo);
    res = dataset;
  }

  return res;
}

DataSet
BlueprintLowOrder::import_uniform_to_explicit(const conduit::Node &n_coords,
                                  Array<int32> &conn,
                                  int32 &n_elems,
                                  std::string &shape)
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

  bool is_2d = true;
  if(n_dims.has_path("k"))
  {
    is_2d = false;
    dims[2] = n_dims["k"].to_int();
  }
  if(is_2d)
  {
    shape = "quad";
  }
  else
  {
    shape = "hex";
  }

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
    if(is_2d)
    {
      detail::logical_index_2d(idx, i, dims);
    }
    else
    {
      detail::logical_index_3d(idx, i, dims);
    }

    Vec<Float,3> point;
    point[0] = origin_x + idx[0] * spacing_x;
    point[1] = origin_y + idx[1] * spacing_y;
    if(is_2d)
    {
      point[2] = 0.f;
    }
    else
    {
      point[2] = origin_z + idx[2] * spacing_z;
    }

    coords_ptr[i] = point;
  }

  Vec<int32,3> cell_dims;
  cell_dims[0] = dims[0] - 1;
  cell_dims[1] = dims[1] - 1;
  n_elems = cell_dims[0] * cell_dims[1];;
  if(!is_2d)
  {
    cell_dims[2] = dims[2] - 1;
    n_elems *= cell_dims[2];
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
      detail::logical_index_2d(idx, i, cell_dims);
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
      detail::logical_index_3d(idx, i, cell_dims);
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
      conn_ptr[offset + 4] = conn_ptr[offset + 0] + dims[0] * dims[1];
      conn_ptr[offset + 5] = conn_ptr[offset + 4] + 1;
      // advance in y
      conn_ptr[offset + 6] = conn_ptr[offset + 4] + dims[0];
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
