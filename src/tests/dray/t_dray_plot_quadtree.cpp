// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)


#include "test_config.h"
#include "gtest/gtest.h"
#include "t_utils.hpp"

#include <set>
#include <numeric>

#include <dray/types.hpp>
#include <dray/vec.hpp>
#include <dray/data_model/data_set.hpp>
#include <dray/uniform_topology.hpp>
#include <dray/uniform_faces.hpp>
#include <dray/data_model/low_order_field.hpp>
#include <dray/quadtree.cpp>
#include <dray/pagani.hpp>

// uniform_cube()
std::shared_ptr<dray::UniformTopology>
uniform_cube(dray::Float spacing,
             dray::int32 cell_dims);

// uniform_stick()
std::shared_ptr<dray::UniformTopology>
uniform_stick(dray::Float spacing,
              dray::int32 cell_dims);

// uniform_face_centers()
dray::Array<dray::FaceLocation> uniform_face_centers(
    const dray::UniformTopology &mesh,
    dray::UniformFaces &face_map);

// uniform_boundary_face_centers()
dray::Array<dray::FaceLocation> uniform_boundary_face_centers(
    const dray::UniformTopology &mesh,
    dray::UniformFaces &face_map);

// append_error()
void append_error(conduit::Node &n_dataset, dray::Array<dray::Float> field);


// Integrand
struct Integrand
{
  const dray::UniformTopology::Evaluator &xyz;
  const dray::UniformTopology::JacobianEvaluator &jac;
  const dray::Vec<dray::Float, 3> source;
  const dray::Float strength;
  const dray::Float extinction;

  DRAY_LAMBDA dray::Float operator()(const dray::FaceLocation &floc) const;
};

// finish()
template <typename Jac, typename XYZ, typename IntegrandT>
void finish(
    dray::PaganiIteration<Jac, IntegrandT> &pagani,
    const XYZ &xyz,
    const IntegrandT &integrand,
    bool output_quadtree,
    const std::string output_file = "");

// for_leafs()
template <typename LeafKernel>
void for_leafs(const dray::QuadTreeForest &forest, const LeafKernel &kernel);

// tree_depth_as_image_dataset()
void tree_depth_as_image_dataset(
    const dray::QuadTreeForest &forest,
    dray::int32 face_id,
    dray::int32 depth,
    conduit::Node &bp_dataset);


//
// dray_qt_challenge
//
TEST(dray_quadtree, dray_qt_challenge)
{
  std::string output_path = prepare_output_dir ();
  std::string output_file =
  conduit::utils::join_file_path (output_path, "qt_challenge");

  using dray::Float;
  using dray::int32;
  using dray::Vec;

  const Float tol_rel = 3e-4;
  const int32 iter_max = 6;
  const int32 nodes_max = 8e4;
  /// const int32 nodes_max = -1;

  std::shared_ptr<dray::UniformTopology> mesh = uniform_stick(1.0, 4);
  dray::DataSet dataset(mesh);

  dray::UniformFaces face_map;
  dray::Array<dray::FaceLocation> face_centers =
      uniform_face_centers(*mesh, face_map);

  const dray::UniformTopology::Evaluator xyz = mesh->evaluator();
  const dray::UniformTopology::JacobianEvaluator jacobian = mesh->jacobian_evaluator();
  const Vec<Float, 3> source = {{-1.0/16, 0.5, 0.5}};
  const Float strength = 13;  // arbitrary point source term
  const Float extinction = 0.10;  // arbitrary absorption coefficient
  const Integrand integrand = Integrand{xyz, jacobian, source, strength, extinction};

  // PaganiIteration<Jacobian, Integrand>
  auto pagani = dray::pagani_iteration(
      face_centers, jacobian, integrand, tol_rel, nodes_max, iter_max );

  pagani.printing(false);
  const bool output_quadtree = true;
  remove_test_file(output_file);
  finish(pagani, xyz, integrand, output_quadtree, output_file);
  const dray::QuadTreeForest &forest = pagani.forest();
  dray::DeviceQuadTreeForest d_forest(forest);

  RAJA::ReduceMax<dray::reduce_policy, int32> max_tree_depth(0);
  for_leafs(forest, [=] DRAY_LAMBDA (int32 i, int32 leaf) {
      max_tree_depth.max(d_forest.quadrant(leaf).depth());
  });

  const int32 depth = max_tree_depth.get(); ///max(max_tree_depth.get(), 5);
  fprintf(stdout, "max tree depth is %d, using %d.\n",
      max_tree_depth.get(), depth);

  std::vector<int32> faces(face_centers.size());
  std::iota(faces.begin(), faces.end(), 0);
  for (const int32 face_id : faces)
  {
    conduit::Node bp_dataset;
    tree_depth_as_image_dataset(forest, face_id, depth, bp_dataset);

    char cycle_suffix[8] = "_000000";
    snprintf(cycle_suffix, 8, "_%06d", face_id);
    const std::string ext = ".blueprint_root";  // visit sees time series if use json format.
    conduit::relay::io::blueprint::save_mesh(
        bp_dataset,
        output_file + "-tree-depth" + std::string(cycle_suffix) + ext);
  }

  // 'Right' answer based on setting rel_tol=1e-6 and max_nodes=-1
  EXPECT_NEAR(149.785934, pagani.value_error().value(), 0.05);
}


namespace detail
{
  DRAY_EXEC dray::int32 pixel_tree_depth(
      const dray::DeviceQuadTreeForest &d_forest,
      dray::int32 tree_id,
      dray::Vec<dray::Float, 2> xy)
  {
    dray::TreeNodePtr &node = tree_id;
    dray::int32 depth = 0;
    while (!d_forest.leaf(node))
    {
      xy *= 2;
      const bool x = int32(xy[0]), y = int32(xy[1]);
      xy[0] -= x;
      xy[1] -= y;

      const int32 child_num = (x << 0) + (y << 1);
      node = d_forest.child(node, child_num);
      depth++;
    }
    return depth;
  }
}

void tree_depth_as_image_dataset(
    const dray::QuadTreeForest &forest,
    dray::int32 face_id,
    dray::int32 depth,
    conduit::Node &bp_dataset)
{
  using namespace dray;
  DeviceQuadTreeForest d_forest(forest);
  const int32 tree_id = face_id;
  const int32 resolution = 1 << depth;

  Vec<Float, 3> origin = {{0, 0, 0}};
  Vec<Float, 3> vspacing = {{Float(1)/resolution, Float(1)/resolution, 0}};
  Vec<int32, 3> vcell_dims = {{resolution, resolution, 1}};
  std::shared_ptr<UniformTopology> mesh =
      std::make_shared<UniformTopology>(vspacing, origin, vcell_dims);
  mesh->name("topo");
  DataSet dataset(mesh);

  UniformTopology::Evaluator eval = mesh->evaluator();

  const int32 num_pixels = mesh->cells();
  Array<Float> tree_depth;
  tree_depth.resize(num_pixels);
  NonConstDeviceArray<Float> d_tree_depth(tree_depth);

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, num_pixels),
      [=] DRAY_LAMBDA (int32 i)
  {
    const Vec<Float, 3> xyz = eval({i, {{.5,.5,.5}}});
    const Vec<Float, 2> xy = {{xyz[0], xyz[1]}};
    const int32 result_depth = ::detail::pixel_tree_depth(d_forest, tree_id, xy);
    d_tree_depth.get_item(i) = result_depth;
  });

  std::shared_ptr<LowOrderField> field =
      std::make_shared<LowOrderField>(tree_depth, LowOrderField::Assoc::Element, vcell_dims);
  field->name("tree_depth");
  dataset.add_field(field);

  dataset.to_blueprint(bp_dataset);

  conduit::Node verify_info;
  if(!conduit::blueprint::mesh::verify(bp_dataset, verify_info))
  {
    std::cout << "Verify failed!" << std::endl;
    verify_info.print();
  }
}


// for_leafs()
template <typename LeafKernel>
void for_leafs(const dray::QuadTreeForest &forest, const LeafKernel &kernel)
{
  const dray::int32 num_leafs = forest.num_leafs();
  dray::ConstDeviceArray<dray::TreeNodePtr> d_leafs(forest.leafs());
  dray::DeviceQuadTreeForest d_forest(forest);
  RAJA::forall<dray::for_policy>(RAJA::RangeSegment(0, num_leafs), [=,&kernel] DRAY_LAMBDA
      (dray::int32 i)
  {
    const dray::TreeNodePtr leaf = d_leafs.get_item(i);
    kernel(i, leaf);
  });
}


// Integrand::()
DRAY_LAMBDA dray::Float Integrand::operator()(const dray::FaceLocation &floc) const
{
  using namespace dray;

  Vec<Float, 3> face_normal = floc.world_normal(jac(floc.loc()));

  const Vec<Float, 3> r = xyz(floc.loc()) - source;
  const Float mag2 = r.magnitude2(),  mag = sqrt(mag2);

  if (dot(face_normal, r) < 0)
    face_normal = -face_normal;

  const Vec<Float, 3> field =
      r.normalized() * (strength / mag2 * exp(-mag * extinction));

  const Float grand = dot(field, face_normal);
  return grand;
}


// finish()
template <typename Jac, typename XYZ, typename IntegrandT>
void finish(
    dray::PaganiIteration<Jac, IntegrandT> &pagani,
    const XYZ &xyz,
    const IntegrandT &integrand,
    bool output_quadtree,
    const std::string output_file)
{
  conduit::Node bp_dataset;
  const std::string extension = ".blueprint_root";  // visit sees time series if use json format.
  char cycle_suffix[8] = "_000000";

  if (output_quadtree)
  {
    pagani.forest().to_blueprint(pagani.m_face_centers, xyz, integrand, bp_dataset);
    conduit::relay::io::blueprint::save_mesh(bp_dataset, output_file + std::string(cycle_suffix) + extension);
  }

  for (int32 level = 1; pagani.need_more_refinements(); ++level)
  {
    pagani.execute_refinements();

    if (output_quadtree)
    {
      snprintf(cycle_suffix, 8, "_%06d", level);
      pagani.forest().to_blueprint(pagani.m_face_centers, xyz, integrand, bp_dataset);
      conduit::relay::io::blueprint::save_mesh(bp_dataset, output_file + std::string(cycle_suffix) + extension);
    }
  }
}


// append_error()
void append_error(conduit::Node &n_dataset, dray::Array<dray::Float> field)
{
  Node &n_error_field = n_dataset["fields/avg_error"];
  n_error_field["association"] = "element";
  n_error_field["topology"] = "topo";
  n_error_field["values"].set_external(field.get_host_ptr(), field.size());
}



//
// uniform_cube()
//
std::shared_ptr<dray::UniformTopology>
uniform_cube(dray::Float spacing,
             dray::int32 cell_dims)
{
  dray::Vec<dray::Float, 3> origin = {{0, 0, 0}};
  dray::Vec<dray::Float, 3> vspacing = {{spacing, spacing, spacing}};
  dray::Vec<dray::int32, 3> vcell_dims = {{cell_dims, cell_dims, cell_dims}};

  return std::make_shared<dray::UniformTopology>
      (vspacing, origin, vcell_dims);
}

//
// uniform_stick()
//
std::shared_ptr<dray::UniformTopology>
uniform_stick(dray::Float spacing,
              dray::int32 cell_dims)
{
  dray::Vec<dray::Float, 3> origin = {{0, 0, 0}};
  dray::Vec<dray::Float, 3> vspacing = {{spacing, spacing, spacing}};
  dray::Vec<dray::int32, 3> vcell_dims = {{cell_dims, 1, 1}};

  return std::make_shared<dray::UniformTopology>
      (vspacing, origin, vcell_dims);
}


//
// uniform_face_centers()
//
dray::Array<dray::FaceLocation> uniform_face_centers(
    const dray::UniformTopology &mesh,
    dray::UniformFaces &face_map)
{
  dray::Array<dray::FaceLocation> face_centers;

  face_map = dray::UniformFaces::from_uniform_topo( mesh);
  face_centers.resize(face_map.num_total_faces());
  face_map.fill_total_faces(face_centers.get_host_ptr());

  return face_centers;
}

//
// uniform_boundary_face_centers()
//
dray::Array<dray::FaceLocation> uniform_boundary_face_centers(
    const dray::UniformTopology &mesh,
    dray::UniformFaces &face_map)
{
  dray::Array<dray::FaceLocation> face_centers;

  face_map = dray::UniformFaces::from_uniform_topo( mesh);
  face_centers.resize(face_map.num_boundary_faces());
  face_map.fill_boundary_faces(face_centers.get_host_ptr());

  return face_centers;
}
