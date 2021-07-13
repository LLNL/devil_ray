// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "t_utils.hpp"
#include "test_config.h"
#include "gtest/gtest.h"

#include <conduit_blueprint.hpp>
#include <conduit_relay.hpp>

#include <RAJA/RAJA.hpp>
#include <dray/policies.hpp>
#include <dray/exports.hpp>
#include <dray/array_utils.hpp>
#include <dray/io/blueprint_reader.hpp>
#include <dray/filters/first_scatter.hpp>
#include <dray/data_model/data_set.hpp>
#include <dray/data_model/collection.hpp>
#include <dray/uniform_topology.hpp>
#include <dray/data_model/low_order_field.hpp>
#include <dray/rendering/framebuffer.hpp>
#include <dray/rendering/device_framebuffer.hpp>


std::shared_ptr<dray::UniformTopology>
uniform_cube(const dray::Vec<dray::Float, 3> &v_origin,
             dray::Float spacing,
             dray::int32 cell_dims);

dray::Collection two_domains(dray::int32 resolution);

dray::UniformTopology * to_uniform(dray::Mesh *mesh);


class OpaqueBlocker
{
  private:
    dray::Vec<dray::Float, 3> m_min;
    dray::Vec<dray::Float, 3> m_max;
    DRAY_EXEC OpaqueBlocker();

  public:
    DRAY_EXEC OpaqueBlocker(
      const dray::Vec<dray::Float, 3> &min,
      const dray::Vec<dray::Float, 3> &max);

    DRAY_EXEC OpaqueBlocker(const OpaqueBlocker &) = default;
    DRAY_EXEC OpaqueBlocker(OpaqueBlocker &&) = default;
    DRAY_EXEC OpaqueBlocker & operator=(const OpaqueBlocker &) = default;
    DRAY_EXEC OpaqueBlocker & operator=(OpaqueBlocker &&) = default;

    DRAY_EXEC bool visibility(const dray::Ray &ray) const;
};


class ImagePlane
{
  private:
    dray::Vec<dray::Float, 3> m_origin;
    dray::Vec<dray::Float, 3> m_spacing;
    dray::Vec<dray::int32, 3> m_cell_dims;
  public:
    ImagePlane(
        const dray::Vec<dray::Float, 3> &m_origin,
        const dray::Vec<dray::Float, 3> &m_spacing,
        const dray::Vec<dray::int32, 3> &m_cell_dims);

    dray::Array<dray::Vec<dray::Float, 3>> world_pixels(
        dray::int32 normal_axis,
        dray::int32 width,
        dray::int32 height) const;
};


//
// dray_shadow
//
TEST (dray_shadow, dray_shadow)
{
  std::string output_path = prepare_output_dir ();
  std::string output_file =
  conduit::utils::join_file_path (output_path, "shadow_plane");
  std::string output_file_mid = output_file + "_mid";
  std::string output_file_back = output_file + "_back";
  remove_test_image (output_file_mid);
  remove_test_image (output_file_back);
  remove_test_image (output_file_mid + "_interp");
  remove_test_image (output_file_back + "_interp");
  remove_test_file (output_file_mid + ".blueprint_root_hdf5.root");
  remove_test_file (output_file_back + ".blueprint_root_hdf5.root");
  remove_test_file (output_file_mid + "_interp" + ".blueprint_root_hdf5");
  remove_test_file (output_file_back + "_interp" + ".blueprint_root_hdf5");

  // Box with dimensions 2x1x1, resolution 32x16x16.
  const int resolution = 16;
  dray::Collection collection = two_domains(resolution);

  OpaqueBlocker blocker({{0.50f, 0.45f, 0.45}},
                        {{0.75f, 0.55f, 0.55}});

  const dray::Vec<dray::Float, 3> source = {{0, 0.5f, 0.5f}};
  const dray::Float strength = 1;

  dray::UniformTopology *mesh0 = to_uniform(collection.domain(0).mesh());
  dray::UniformTopology *mesh1 = to_uniform(collection.domain(1).mesh());

  const dray::Vec<dray::Float, 3> one_x = {{1, 0, 0}};
  ImagePlane plane_mid(mesh0->origin() + one_x, mesh0->spacing(), mesh0->cell_dims());
  ImagePlane plane_back(mesh1->origin() + one_x, mesh1->spacing(), mesh1->cell_dims());
  dray::Array<dray::Vec<dray::Float, 3>> pixel_positions;
  dray::Vec<dray::Float, 3> *pixel_positions_ptr;

  const dray::int32 width = 512;
  const dray::int32 height = 512;

  dray::Framebuffer frame_buffer(width, height);
  conduit::Node conduit_frame_buffer;

  // -------------------------
  // Ground truth
  // -------------------------

  auto ground_truth = [=] DRAY_LAMBDA
      (const dray::Vec<dray::Float, 3> &pos)
  {
    dray::Ray ray;
    ray.m_dir = pos - source;
    ray.m_orig = source;

    const bool visible = blocker.visibility(ray);
    return (visible ? 0.0f : 1024.0f);
  };

  frame_buffer.clear({{0, 0, 0, 0}});
  dray::DeviceFramebuffer dvc_frame_buffer(frame_buffer);
  pixel_positions = plane_mid.world_pixels(0, width, height);
  pixel_positions_ptr = pixel_positions.get_device_ptr();
  RAJA::forall<dray::for_policy>(RAJA::RangeSegment(0, width * height),
      [=, &dvc_frame_buffer] DRAY_LAMBDA (dray::int32 pixel)
  {
    const dray::Vec<dray::Float, 3> position = pixel_positions_ptr[pixel];
    const dray::Float sigma_t = ground_truth(position);
    const dray::Float scalar = exp(-sigma_t);
    dvc_frame_buffer.set_color(pixel, {{scalar, scalar, scalar, 1}});
    dvc_frame_buffer.set_depth(pixel, scalar);
  });
  frame_buffer.save(output_file_mid);
  frame_buffer.to_node(conduit_frame_buffer);
  conduit::relay::io::blueprint::save_mesh(conduit_frame_buffer, output_file_mid + ".blueprint_root_hdf5");

  frame_buffer.clear({{0, 0, 0, 0}});
  dvc_frame_buffer = dray::DeviceFramebuffer(frame_buffer);
  pixel_positions = plane_back.world_pixels(0, width, height);
  pixel_positions_ptr = pixel_positions.get_device_ptr();
  RAJA::forall<dray::for_policy>(RAJA::RangeSegment(0, width * height),
      [=, &dvc_frame_buffer] DRAY_LAMBDA (dray::int32 pixel)
  {
    const dray::Vec<dray::Float, 3> position = pixel_positions_ptr[pixel];
    const dray::Float sigma_t = ground_truth(position);
    const dray::Float scalar = exp(-sigma_t);
    dvc_frame_buffer.set_color(pixel, {{scalar, scalar, scalar, 1}});
    dvc_frame_buffer.set_depth(pixel, scalar);
  });
  frame_buffer.save(output_file_back);
  frame_buffer.to_node(conduit_frame_buffer);
  conduit::relay::io::blueprint::save_mesh(conduit_frame_buffer, output_file_back + ".blueprint_root_hdf5");



  // -------------------------
  // Interpolation surfaces
  // -------------------------

  // Store ground truth at I.S. vertices.
  const dray::UniformTopology::Evaluator xyz0 = mesh0->evaluator();
  dray::Array<dray::Float> vert_field_0;
  vert_field_0.resize((resolution+1) * (resolution+1) * (resolution+1));
  dray::array_memset_zero(vert_field_0);
  dray::Float *vert_field_0_ptr = vert_field_0.get_device_ptr();
  RAJA::forall<dray::for_policy>(RAJA::RangeSegment(0, (resolution+1)*(resolution+1)),
      [=] DRAY_LAMBDA (dray::int32 vert_idx)
  {
    const dray::int32 i = vert_idx % (resolution + 1);
    const dray::int32 j = vert_idx / (resolution + 1);
    const dray::int32 i_clamp = (i == resolution ? resolution - 1: i);
    const dray::int32 j_clamp = (j == resolution ? resolution - 1: j);
    dray::Location loc;
    loc.m_cell_id = j_clamp * resolution * resolution + i_clamp * resolution + (resolution - 1);
    loc.m_ref_pt[0] = 1.0f;
    loc.m_ref_pt[1] = (i == resolution);
    loc.m_ref_pt[2] = (j == resolution);

    const dray::Vec<dray::Float, 3> world_pt = xyz0(loc);
    const dray::Float sigma_t = ground_truth(world_pt);
    const dray::int32 vert_id = ((j * (resolution+1)) + i) * (resolution+1) + resolution;
    vert_field_0_ptr[vert_id] = sigma_t;
  });
  dray::LowOrderField field_0(vert_field_0, dray::LowOrderField::Assoc::Vertex, mesh0->cell_dims());

  // Output I.S. to image.
  frame_buffer.clear({{0, 0, 0, 0}});
  dvc_frame_buffer = dray::DeviceFramebuffer(frame_buffer);
  pixel_positions = plane_mid.world_pixels(0, width, height);
  dray::Array<dray::Location> pixel_locations_0 = mesh0->locate(pixel_positions);
  dray::Array<dray::Float> pixel_scalars;
  field_0.eval(pixel_locations_0, pixel_scalars);
  const dray::Float * pixel_scalars_ptr = pixel_scalars.get_device_ptr_const();
  RAJA::forall<dray::for_policy>(RAJA::RangeSegment(0, width * height),
      [=, &dvc_frame_buffer] DRAY_LAMBDA (dray::int32 pixel)
  {
    const dray::Float sigma_t = pixel_scalars_ptr[pixel];
    const dray::Float scalar = exp(-sigma_t);
    dvc_frame_buffer.set_color(pixel, {{scalar, scalar, scalar, 1}});
    dvc_frame_buffer.set_depth(pixel, scalar);
  });
  frame_buffer.save(output_file_mid + "_interp");
  frame_buffer.to_node(conduit_frame_buffer);
  conduit::relay::io::blueprint::save_mesh(
      conduit_frame_buffer, output_file_mid + "_interp" + ".blueprint_root_hdf5");

  // Transfer I.S. from domain 0 to domain 1.
  dray::Array<dray::Float> vert_field_1;
  vert_field_1.resize((resolution+1) * (resolution+1) * (resolution+1));
  dray::array_memset_zero(vert_field_1);
  dray::Float *vert_field_1_ptr = vert_field_1.get_device_ptr();
  RAJA::forall<dray::for_policy>(RAJA::RangeSegment(0, (resolution+1)*(resolution+1)),
      [=] DRAY_LAMBDA (dray::int32 vert_idx)
  {
    const dray::int32 i = vert_idx % (resolution + 1);
    const dray::int32 j = vert_idx / (resolution + 1);
    const dray::int32 vert_id_0 = ((j * (resolution+1)) + i) * (resolution+1) + resolution;
    const dray::int32 vert_id_1 = ((j * (resolution+1)) + i) * (resolution+1) + 0;
    vert_field_1_ptr[vert_id_1] = vert_field_0_ptr[vert_id_0];
  });
  dray::LowOrderField field_1(vert_field_1, dray::LowOrderField::Assoc::Vertex, mesh1->cell_dims());

  // Trace to back plane using I.S.
  frame_buffer.clear({{0, 0, 0, 0}});
  dvc_frame_buffer = dray::DeviceFramebuffer(frame_buffer);
  pixel_positions = plane_back.world_pixels(0, width, height);
  pixel_positions_ptr = pixel_positions.get_device_ptr();
  RAJA::forall<dray::for_policy>(RAJA::RangeSegment(0, width * height),
      [=] DRAY_LAMBDA (dray::int32 pixel)
  {
    const dray::Vec<dray::Float, 3> endpoint = pixel_positions_ptr[pixel];
    const dray::Vec<dray::Float, 3> ray = endpoint - source;
    // Intersect at x=1
    dray::Float t = 1.0f / ray[0];
    const dray::Vec<dray::Float, 3> intersection = source + ray * t;
    pixel_positions_ptr[pixel] = intersection;
  });
  // Eval.
  dray::Array<dray::Location> pixel_locations_1 = mesh1->locate(pixel_positions);
  field_1.eval(pixel_locations_1, pixel_scalars);
  pixel_scalars_ptr = pixel_scalars.get_device_ptr_const();
  // Save to image.
  RAJA::forall<dray::for_policy>(RAJA::RangeSegment(0, width * height),
      [=, &dvc_frame_buffer] DRAY_LAMBDA (dray::int32 pixel)
  {
    const dray::Float sigma_t = pixel_scalars_ptr[pixel];
    const dray::Float scalar = exp(-sigma_t);
    dvc_frame_buffer.set_color(pixel, {{scalar, scalar, scalar, 1}});
    dvc_frame_buffer.set_depth(pixel, scalar);
  });
  frame_buffer.save(output_file_back + "_interp");
  frame_buffer.to_node(conduit_frame_buffer);
  conduit::relay::io::blueprint::save_mesh(
      conduit_frame_buffer, output_file_back + "_interp" + ".blueprint_root_hdf5");
}




// OpaqueBlocker()
DRAY_EXEC OpaqueBlocker::OpaqueBlocker()
  : m_min({{0, 0, 0}}),
    m_max({{1, 1, 1}})
{}

// OpaqueBlocker()
DRAY_EXEC OpaqueBlocker::OpaqueBlocker(
  const dray::Vec<dray::Float, 3> &min,
  const dray::Vec<dray::Float, 3> &max)
  : m_min(min),
    m_max(max)
{}

// OpaqueBlocker::visibility()
DRAY_EXEC bool OpaqueBlocker::visibility(const dray::Ray &ray) const
{
  dray::Range t_range = dray::Range::mult_identity();  // intesections
  for (dray::int32 d = 0; d < 3; ++d)
  {
    const dray::Float t_0 = (m_min[d] - ray.m_orig[d]) / ray.m_dir[d];
    const dray::Float t_1 = (m_max[d] - ray.m_orig[d]) / ray.m_dir[d];
    dray::Range range_i = dray::Range::identity();  // unions
    range_i.include(t_0);
    range_i.include(t_1);
    t_range = t_range.intersect(range_i);
  }
  return t_range.is_empty();
}

// ImagePlane()
ImagePlane::ImagePlane(
    const dray::Vec<dray::Float, 3> &origin,
    const dray::Vec<dray::Float, 3> &spacing,
    const dray::Vec<dray::int32, 3> &cell_dims)
  : m_origin(origin),
    m_spacing(spacing),
    m_cell_dims(cell_dims)
{}

// ImagePlane::world_pixels()
dray::Array<dray::Vec<dray::Float, 3>> ImagePlane::world_pixels(
    dray::int32 normal_axis,
    dray::int32 width,
    dray::int32 height) const
{
  using namespace dray;
  const int32 u_axis = (normal_axis != 0 ? 0 : 1);
  const int32 v_axis = (normal_axis != 2 ? 2 : 1);
  Vec<Float, 3> du = {{0, 0, 0}};
  Vec<Float, 3> dv = {{0, 0, 0}};
  du[u_axis] = m_spacing[u_axis] * m_cell_dims[u_axis];
  dv[v_axis] = m_spacing[v_axis] * m_cell_dims[v_axis];

  const Vec<Float, 3> origin = m_origin;

  const int32 pixels = width * height;

  Array<Vec<Float, 3>> pixel_positions;
  pixel_positions.resize(pixels);
  Vec<Float, 3> * pixel_positions_ptr = pixel_positions.get_device_ptr();

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, pixels),
      [=] DRAY_LAMBDA (int32 pixel)
  {
    const int32 i = pixel % height;
    const int32 j = pixel / height;
    const Float u = Float(i + 0.5) / width;
    const Float v = Float(j + 0.5) / height;
    const Vec<Float, 3> position = origin + du * u + dv * v;
    pixel_positions_ptr[pixel] = position;
  });

  return pixel_positions;
}


// uniform_cube()
std::shared_ptr<dray::UniformTopology>
uniform_cube(const dray::Vec<dray::Float, 3> &v_origin,
             dray::Float spacing,
             dray::int32 cell_dims)
{
  dray::Vec<dray::Float, 3> v_spacing = {{spacing, spacing, spacing}};
  dray::Vec<dray::int32, 3> v_cell_dims = {{cell_dims, cell_dims, cell_dims}};

  return std::make_shared<dray::UniformTopology>
      (v_spacing, v_origin, v_cell_dims);
}

// to_uniform()
dray::UniformTopology * to_uniform(dray::Mesh *mesh)
{
  return dynamic_cast<dray::UniformTopology *>(mesh);
}

// two_domains()
dray::Collection two_domains(dray::int32 resolution)
{
  using namespace dray;
  Collection collection;

  collection.add_domain(DataSet(uniform_cube(
          Vec<Float, 3>({{0, 0, 0}}),
          1./resolution,
          resolution)));
  collection.add_domain(DataSet(uniform_cube(
          Vec<Float, 3>({{1, 0, 0}}),
          1./resolution,
          resolution)));

  return collection;
}
