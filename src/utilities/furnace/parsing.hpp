// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef FURNACE_PARSING_HPP
#define FURNACE_PARSING_HPP

#include <dray/rendering/camera.hpp>
#include <dray/dray.hpp>
#include <dray/collection.hpp>
#include <dray/io/blueprint_reader.hpp>
#include <dray/import_order_policy.hpp>

#ifdef MPI_ENABLED
#include <mpi.h>
#endif

void init_furnace()
{
#ifdef MPI_ENABLED
  MPI_Init(nullptr, nullptr);
  MPI_Comm comm = MPI_COMM_WORLD;
  dray::dray::mpi_comm(MPI_Comm_c2f(comm));
#endif
}

void finalize_furnace()
{
#ifdef MPI_ENABLED
  MPI_Finalize();
#endif
}

void print_fields (dray::Collection &collection)
{
  dray::DataSet dataset = collection.domain(0);
  std::vector<std::string> fields = dataset.fields();
  for (int i = 0; i < fields.size(); ++i)
  {
    std::cerr << "'" << fields[i] << "'\n";
  }
}

void parse_camera (dray::Camera &camera, const conduit::Node &camera_node)
{
  typedef dray::Vec<float, 3> Vec3f;
  //
  // Get the optional camera parameters
  //
  if (camera_node.has_child ("look_at"))
  {
    conduit::Node n;
    camera_node["look_at"].to_float64_array (n);
    const conduit::float64 *coords = n.as_float64_ptr ();
    Vec3f look_at{ float (coords[0]), float (coords[1]), float (coords[2]) };
    camera.set_look_at (look_at);
  }

  if (camera_node.has_child ("position"))
  {
    conduit::Node n;
    camera_node["position"].to_float64_array (n);
    const conduit::float64 *coords = n.as_float64_ptr ();
    Vec3f position{ float (coords[0]), float (coords[1]), float (coords[2]) };
    camera.set_pos (position);
  }

  if (camera_node.has_child ("up"))
  {
    conduit::Node n;
    camera_node["up"].to_float64_array (n);
    const conduit::float64 *coords = n.as_float64_ptr ();
    Vec3f up{ float (coords[0]), float (coords[1]), float (coords[2]) };
    up.normalize ();
    camera.set_up (up);
  }

  if (camera_node.has_child ("fov"))
  {
    camera.set_fov (camera_node["fov"].to_float64 ());
  }

  if (camera_node.has_child ("width"))
  {
    camera.set_width (camera_node["width"].to_int32 ());
  }

  if (camera_node.has_child ("height"))
  {
    camera.set_height (camera_node["height"].to_int32 ());
  }

  // this is an offset from the current azimuth
  if (camera_node.has_child ("azimuth"))
  {
    double azimuth = camera_node["azimuth"].to_float64 ();
    camera.azimuth (azimuth);
  }
  if (camera_node.has_child ("elevation"))
  {
    double elevation = camera_node["elevation"].to_float64 ();
    camera.elevate (elevation);
  }
}

struct Config
{
  std::string m_file_name;
  conduit::Node m_config;
  dray::Collection m_collection;
  dray::Camera m_camera;
  std::string m_field;
  int m_trials;

  Config () = delete;

  Config (std::string config_file)
  {
    m_config.load (config_file, "yaml");
  }

  void load_data ()
  {
    if (!m_config.has_path ("root_file"))
    {
      throw std::runtime_error ("missing 'root_file'");
    }
    std::string root_file = m_config["root_file"].as_string ();

    // Default order policy.
    bool use_fixed_mesh_order = true;
    bool use_fixed_field_order = true;

    // Load order policy.
    if (m_config.has_path("use_fixed_mesh_order"))
    {
      use_fixed_mesh_order = (m_config["use_fixed_mesh_order"].as_string() != "true");
    }
    if (m_config.has_path("use_fixed_field_order"))
    {
      use_fixed_field_order = (m_config["use_fixed_field_order"].as_string() != "true");
    }

    dray::dray::prefer_native_order_mesh(use_fixed_mesh_order);
    dray::dray::prefer_native_order_field(use_fixed_field_order);
    m_collection = dray::BlueprintReader::load (root_file);
  }

  void load_field ()
  {
    if (!m_config.has_path ("field"))
    {
      throw std::runtime_error ("missing 'field'");
    }

    m_field = m_config["field"].as_string ();
    if (!m_collection.has_field (m_field))
    {
      std::cerr << "No field named '" << m_field << "'. Known fields: \n";
      print_fields (m_collection);
      throw std::runtime_error ("bad 'field'");
    }
  }

  void load_camera ()
  {
    // setup a default camera
    m_camera.set_width (1024);
    m_camera.set_height (1024);
    m_camera.reset_to_bounds (m_collection.bounds());

    if (m_config.has_path ("camera"))
    {
      parse_camera (m_camera, m_config["camera"]);
    }
  }
};
#endif
