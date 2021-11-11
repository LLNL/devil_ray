// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <dray/dray.hpp>
#include <dray/utils/appstats.hpp>
#include <dray/utils/png_encoder.hpp>

#include <dray/graph/cpu_graph.hpp>
#include <dray/graph/topological_order.hpp>

#include <dray/uniform_topology.hpp>
#include <dray/data_model/low_order_field.hpp>
#include <dray/data_model/data_set.hpp>
#include <dray/data_model/collection.hpp>
#include <dray/uniform_indexer.hpp>

#include <dray/transport/uniform_partials.hpp>
#include <dray/transport/compose_domain_segment.hpp>
#include <dray/host_array.hpp>
#include <dray/array_utils.hpp>
#include <RAJA/RAJA.hpp>

#include "parsing.hpp"
#include <conduit.hpp>
#include <conduit_blueprint.hpp>
#include <conduit_relay.hpp>
#include <iostream>
#include <random>


dray::Collection egg_cartons(
    const std::string &field_name,
    dray::float64 sigmat_amplitude,
    dray::float64 sigmat_period,
    const dray::Vec<dray::Float, 3> &origin,
    const dray::Vec<dray::Float, 3> &spacing,
    const dray::Vec<dray::int32, 3> &domains_layout,
    const dray::Vec<dray::int32, 3> &cells_per_domain);


dray::Collection uniform_absorption(
    const std::string &field_name,
    dray::float64 absorption,
    const dray::Vec<dray::Float, 3> &origin,
    const dray::Vec<dray::Float, 3> &spacing,
    const dray::Vec<dray::int32, 3> &domains_layout,
    const dray::Vec<dray::int32, 3> &cells_per_domain);






namespace dray
{

  // Precondition: `nearby_domains` contains all domains
  //               adjacent to any in `rank_domains`

  // for each domain in nearby_domains:
  //   produce panels
  //     all_pairs <- Pair(panel, domain, panel is {upwind/downwind} of domain);
  //     all_panels <- panel

  // separate all_panels by plane (orientation and sheet displacement)

  // Graph upwind;       // (i,j): j is upwind of i
  // Graph downwind;     // (i,j): j is downwind of i

  // within a plane:
  //   Generage list of AABB<2>
  //   build tree
  //   traverse for positive overlap (> cell_res/2):
  //   for each pair of overlapping panels (A, B)
  //           such that (upwind) domA -> A : B -> domB (downwind)
  //     upwind.insert_edge(domB, domA);
  //     downwind.insert_edge(domA, domB);

  //


  // Locally, just the index of a domain in a collection,
  // but can extend globally to refer to a pair (rank, dom_idx)
  struct DomainToken
  {
    bool local = true;
    int32 idx;
    /// int32 rank;

    static DomainToken local_domain(int32 idx)
    {
      DomainToken token;
      token.local = true;
      token.idx = idx;
      ///token.rank = ???
      return token;
    }
  };

  struct DomainSocket
  {
    DomainToken domain;
    int32 socket;
  };

  UniformTopology * structured(Mesh * mesh)
  {
    UniformTopology * structured_mesh = dynamic_cast<UniformTopology *>(mesh);
    if (structured_mesh == nullptr)
      throw std::invalid_argument("mesh is not structured.");
    return structured_mesh;
  }

  LowOrderField * structured(Field * field)
  {
    LowOrderField * structured_field = dynamic_cast<LowOrderField *>(field);
    if (structured_field == nullptr)
      throw std::invalid_argument("field is not structured.");
    return structured_field;
  }


  // Knows about StructuredDomains and assigning panels to faces
  class StructuredDomainPanel
  {
    public:
      enum Side : uint8 { Z0 = UniformIndexer::Z0,
                          Z1 = UniformIndexer::Z1,
                          Y0 = UniformIndexer::Y0,
                          Y1 = UniformIndexer::Y1,
                          X0 = UniformIndexer::X0,
                          X1 = UniformIndexer::X1,
                          NUM_SIDES = UniformIndexer::NUM_SIDES };

      static UniformIndexer::Side uiside(Side side) {
        return static_cast<UniformIndexer::Side>(side);
      }

    protected:
      int32 m_domain_idx = 0;
      int32 m_socket = 0;
      Vec<Float, 3> m_origin = {{0,0,0}};
      Vec<Float, 3> m_spacing = {{1,1,1}};
      Vec<int32, 3> m_cell_dims = {{1,1,1}};
      Vec<uint8, 3> m_axis_subset = {{1,1,0}};
      Side m_side = Z0;

      friend class PanelList;

    public:
      StructuredDomainPanel()
      {}

      StructuredDomainPanel(
          const UniformTopology *domain_mesh,
          int32 domain_idx,
          int32 socket);

      int32 domain_idx() const;
      int32 socket() const;

      Vec<Float, 3> origin() const { return m_origin; }
      Vec<Float, 3> spacing() const { return m_spacing; }
      Vec<int32, 3> cell_dims() const { return m_cell_dims; }
      Vec<uint8, 3> axis_subset() const { return m_axis_subset; }
      Side side() const { return m_side; }
      UniformIndexer::Side uiside() const { return uiside(m_side); }

      bool upwind_of_domain(const Vec<Float, 3> &source) const;
      bool downwind_of_domain(const Vec<Float, 3> &source) const;

      // If two panels overlap, then these properties match on both panels.
      int32 orientation_code() const;
      Float displacement_code() const;

      // Within a plane, overlap can be tested using an AABB.
      AABB<2> in_plane_aabb() const;
  };

  StructuredDomainPanel::StructuredDomainPanel(
      const UniformTopology *domain_mesh,
      int32 domain_idx,
      int32 socket)
    :
      m_domain_idx(domain_idx),
      m_socket(socket),
      m_side(static_cast<Side>(socket)),
      m_spacing(domain_mesh->spacing()),
      m_cell_dims(domain_mesh->cell_dims())
  {
    m_axis_subset = {{true, true, true}};
    if (m_side == X0 || m_side == X1)
      m_axis_subset[0] = false;
    if (m_side == Y0 || m_side == Y1)
      m_axis_subset[1] = false;
    if (m_side == Z0 || m_side == Z1)
      m_axis_subset[2] = false;

    m_origin = domain_mesh->origin();
    if (m_side == X1)
      m_origin[0] += m_spacing[0] * m_cell_dims[0];
    if (m_side == Y1)
      m_origin[1] += m_spacing[1] * m_cell_dims[1];
    if (m_side == Z1)
      m_origin[2] += m_spacing[2] * m_cell_dims[2];
  }

  int32 StructuredDomainPanel::domain_idx() const
  {
    return m_domain_idx;
  }

  int32 StructuredDomainPanel::socket() const
  {
    return m_socket;
  }

  bool StructuredDomainPanel::upwind_of_domain(
      const Vec<Float, 3> &source) const
  {
    // (m_origin - source) dot (outward normal)
    const int32 comp = (m_side == X0 || m_side == X1 ? 0
                      : m_side == Y0 || m_side == Y1 ? 1
                      :                                2);
    Float dot_out = (m_origin[comp] - source[comp]) * m_spacing[comp];
    if (m_side == X0 || m_side == Y0 || m_side == Z0)
      dot_out = -dot_out;  // negative spacing

    return dot_out <= 0.0f;
  }

  bool StructuredDomainPanel::downwind_of_domain(
      const Vec<Float, 3> &source) const
  {
    return !upwind_of_domain(source);
  }

  int32 StructuredDomainPanel::orientation_code() const
  {
    std::bitset<3> bits;
    for (int32 d = 0; d < 3; ++d)
      bits[d] = m_axis_subset[d];
    return int32(bits.to_ulong());
  }

  Float StructuredDomainPanel::displacement_code() const
  {
    const int32 comp = (m_side == X0 || m_side == X1 ? 0
                      : m_side == Y0 || m_side == Y1 ? 1
                      :                                2);
    return m_origin[comp];
  }

  AABB<2> StructuredDomainPanel::in_plane_aabb() const
  {
    Vec<Float, 2> sub_origin = sub_vec<2>(m_origin, m_axis_subset);
    Vec<Float, 2> sub_spacing = sub_vec<2>(m_spacing, m_axis_subset);
    Vec<int32, 2> sub_cell_dims = sub_vec<2>(m_cell_dims, m_axis_subset);
    Vec<Float, 2> sub_diagonal = {{sub_spacing[0] * sub_cell_dims[0],
                                   sub_spacing[1] * sub_cell_dims[1]}};
    AABB<2> aabb;
    aabb.include(sub_origin);
    aabb.include(sub_origin + sub_diagonal);
    return aabb;
  }

  // Assuming they are connected
  bool a_downwind_of_b(
      const StructuredDomainPanel &panel_a,
      const StructuredDomainPanel &panel_b,
      const Vec<Float, 3> &source)
  {
    return panel_b.downwind_of_domain(source) &&
           panel_a.upwind_of_domain(source);

  }

  bool a_upwind_of_b(
      const StructuredDomainPanel &panel_a,
      const StructuredDomainPanel &panel_b,
      const Vec<Float, 3> &source)
  {
    return panel_a.downwind_of_domain(source) &&
           panel_b.upwind_of_domain(source);
  }




  //==================================================================
  class PanelList
  {
    public:
      static int32 num_sockets(const UniformTopology *domain_mesh);
      // valid sockets are [0..num_sockets)

      PanelList();

      void insert(const StructuredDomainPanel &panel);

      std::vector<std::pair<int32, int32>> adjacent_panels() const;
      // returns "unordered" pairs (i,j) with i<j, as adjacency is symmetric.
      // list is sorted lexicographically.

      const StructuredDomainPanel & operator[](int32 i) const;
      const StructuredDomainPanel & operator()(int32 domain_idx, int32 socket) const;
      size_t size() const;

    protected:
      std::vector<StructuredDomainPanel> m_panels;

      using DomainSocketKey = std::pair<int32, int32>;
      std::map<DomainSocketKey, int32> m_lookup;
  };

  //--------------------------------------------------

  PanelList::PanelList()
  { }

  void PanelList::insert(const StructuredDomainPanel &panel)
  {
    const int32 panel_idx = m_panels.size();
    m_lookup.insert({{panel.domain_idx(), panel.socket()}, panel_idx});

    m_panels.push_back(panel);
  }

  int32 PanelList::num_sockets(const UniformTopology *domain_mesh)
  {
    return StructuredDomainPanel::NUM_SIDES;
  }

  const StructuredDomainPanel & PanelList::operator[](int32 i) const
  {
    return m_panels[i];
  }

  const StructuredDomainPanel & PanelList::operator()(int32 domain_idx, int32 socket) const
  {
    return m_panels[m_lookup.at(DomainSocketKey{domain_idx, socket})];
  }

  size_t PanelList::size() const
  {
    return m_panels.size();
  }

  std::vector<std::pair<int32, int32>> PanelList::adjacent_panels() const
  {
    // returns "unordered" pairs (i,j) with i<j, as adjacency is symmetric.
    // list is sorted lexicographically.

    const size_t num_panels = m_panels.size();

    std::vector<int32> original_idx(num_panels);
    std::iota(original_idx.begin(), original_idx.end(), 0);

    using PlaneKey = std::pair<int32, Float>;
    const auto plane_key = [=](const StructuredDomainPanel &panel)
    {
      return PlaneKey({panel.orientation_code(), panel.displacement_code()});
    };

    struct NamedAABB
    {
      int32 name;
      AABB<2> aabb;
    };

    // Partition by planes before testing for intersections.
    std::multimap<PlaneKey, NamedAABB> planes;
    for (int32 panel_idx = 0; panel_idx < num_panels; ++panel_idx)
    {
      const StructuredDomainPanel &panel = m_panels[panel_idx];

      planes.insert({plane_key(panel),
                     NamedAABB{panel_idx, panel.in_plane_aabb()}});
    }

    const auto aabbs_overlap = [=](const AABB<2> &box_a, const AABB<2> &box_b)
    {
      AABB<2> intersection = box_a.intersect(box_b);
      return !intersection.is_empty() &&
             intersection.min_length() > epsilon<Float>();
    };

    // Pairs of overlapping panels, within plane partitions.
    std::vector<std::pair<int32, int32>> overlapping_pairs;
    std::multimap<PlaneKey, NamedAABB>::const_iterator range_begin, range_end;
    for (range_begin = planes.begin();
         range_begin != planes.end();
         range_begin = range_end)
    {
      // Find range_end.
      range_end = range_begin;
      size_t range_size = 0;
      while (range_end != planes.end() &&
             range_end->first == range_begin->first)
        (++range_end, ++range_size);
      // A plane partition is [range_begin, range_end).

      // Test all unique pairs.
      // In the future, can do something fancy with BVHs.
      std::multimap<PlaneKey, NamedAABB>::const_iterator itx, ity;
      for (itx = range_begin; itx != range_end; ++itx)
        for (ity = itx, ++ity; ity != range_end; ++ity)
          if (aabbs_overlap(itx->second.aabb, ity->second.aabb))
            overlapping_pairs.push_back({
                min(itx->second.name, ity->second.name),
                max(itx->second.name, ity->second.name) });
    }

    std::sort(overlapping_pairs.begin(), overlapping_pairs.end());
    return overlapping_pairs;


  }

  //==================================================================



  struct FaceCurrents//future: Rename to something like InterpolationSurface
  {
    // In future, maybe quadtree-based InterpolationSurface here.

    StructuredDomainPanel m_panel;
    Vec<Float, 3> m_source;

    void discretize(
        const UniformTopology * mesh,
        const StructuredDomainPanel & panel,
        const Vec<Float, 3> &source);

    Array<Vec<Float, 3>> sample_points() const;

    Array<Float> interpolate(
        const Array<Float> sigt,
        const Array<Vec<Float, 3>> query_points) const;
  };

  void FaceCurrents::discretize(
      const UniformTopology * mesh,
      const StructuredDomainPanel & panel,
      const Vec<Float, 3> &source)
  {
    m_panel = panel;
    m_source = source;
  }

  Array<Vec<Float, 3>> FaceCurrents::sample_points() const
  {
    // for now just vertices

    Array<Vec<Float, 3>> vertices;
    const Vec<Float, 3> plane_origin = m_panel.origin();
    const Vec<Float, 2> spacing =
        sub_vec<2>(m_panel.spacing(), m_panel.axis_subset());
    const Vec<int32, 2> cell_dims =
        sub_vec<2>(m_panel.cell_dims(), m_panel.axis_subset());

    vertices.resize((cell_dims[0] + 1) * (cell_dims[1] + 1));
    NonConstHostArray<Vec<Float, 3>> h_vertices(vertices);

    int32 index = 0;
    Float uf = 0.0f,  vf = 0.0f;
    using V = Vec<Float, 3>;
    using SDP = StructuredDomainPanel;

    if (m_panel.side() == SDP::Z0 || m_panel.side() == SDP::Z1)
    {
      for (int32 v = 0; vf = v * spacing[1],  v < cell_dims[1] + 1; ++v)
        for (int32 u = 0; uf = u * spacing[0],  u < cell_dims[0] + 1; ++u)
        {
          const Vec<Float, 3> vertex = plane_origin + V{{uf, vf, 0}};
          h_vertices.get_item(index++) = vertex;
        }
    }
    else if (m_panel.side() == SDP::Y0 || m_panel.side() == SDP::Y1)
    {
      for (int32 v = 0; vf = v * spacing[1],  v < cell_dims[1] + 1; ++v)
        for (int32 u = 0; uf = u * spacing[0],  u < cell_dims[0] + 1; ++u)
        {
          const Vec<Float, 3> vertex = plane_origin + V{{uf, 0, vf}};
          h_vertices.get_item(index++) = vertex;
        }
    }
    else//(m_panel.side() == SDP::X0 || m_panel.side() == SDP::X1)
    {
      for (int32 v = 0; vf = v * spacing[1],  v < cell_dims[1] + 1; ++v)
        for (int32 u = 0; uf = u * spacing[0],  u < cell_dims[0] + 1; ++u)
        {
          const Vec<Float, 3> vertex = plane_origin + V{{0, uf, vf}};
          h_vertices.get_item(index++) = vertex;
        }
    }

    return vertices;
  }

  Array<Float> FaceCurrents::interpolate(
      const Array<Float> sigt,
      const Array<Vec<Float, 3>> query_points) const
  {
    const int32 size = query_points.size();
    if (size == 0)
      return Array<Float>();
    if (sigt.size() == 0)
      fprintf(stderr, "sigt is empty!!!!!!!!\n");

    Array<Float> interp_sigt;
    interp_sigt.resize(size);

    ConstDeviceArray<Vec<Float, 3>> d_query_points(query_points);
    ConstDeviceArray<Float> d_sigt(sigt);
    NonConstDeviceArray<Float> d_interp_sigt(interp_sigt);

    const Vec<uint8, 3> sub_axes = this->m_panel.axis_subset();
    const Vec<Float, 2> origin = sub_vec<2>(this->m_panel.origin(), sub_axes);
    const Vec<Float, 2> spacing = sub_vec<2>(this->m_panel.spacing(), sub_axes);
    const Vec<int32, 2> cell_dims = sub_vec<2>(this->m_panel.cell_dims(), sub_axes);

    // since in sample_points() on the opposing domain we returned the vertices,
    // we know to index sigt as a vertex array

    RAJA::forall<for_policy>(RAJA::RangeSegment(0, size),
        [=] DRAY_LAMBDA (int32 i)
    {
      const Vec<Float, 3> point_3d = d_query_points.get_item(i);
      const Vec<Float, 2> point_2d = sub_vec<2>(point_3d, sub_axes);

      // Integer [0, dims) and fractional part [0, 1]
      Vec<Float, 2> ref = point_2d - origin;
      Vec<int32, 2> cell;
      ref[0] /= spacing[0];
      ref[1] /= spacing[1];
      cell[0] = (int32) ref[0];
      cell[1] = (int32) ref[1];
      if (cell[0] >= cell_dims[0])    // Clamp
        cell[0] = cell_dims[0] - 1;
      if (cell[1] >= cell_dims[1])    // Clamp
        cell[1] = cell_dims[1] - 1;
      ref[0] -= cell[0];
      ref[1] -= cell[1];

      // Four corners
      const int32 i00 = cell[0] +     (cell_dims[1] + 1) * cell[1];
      const int32 i01 = cell[0] + 1 + (cell_dims[1] + 1) * cell[1];
      const int32 i10 = cell[0] +     (cell_dims[1] + 1) * (cell[1] + 1);
      const int32 i11 = cell[0] + 1 + (cell_dims[1] + 1) * (cell[1] + 1);
      const Float f00 = d_sigt.get_item(i00);
      const Float f01 = d_sigt.get_item(i01);
      const Float f10 = d_sigt.get_item(i10);
      const Float f11 = d_sigt.get_item(i11);

      // Bilinear interpolation
      const Float interp_val =
          f00 * (1 - ref[0]) * (1 - ref[1]) +
          f01 * (    ref[0]) * (1 - ref[1]) +
          f10 * (1 - ref[0]) * (    ref[1]) +
          f11 * (    ref[0]) * (    ref[1]);

      d_interp_sigt.get_item(i) = interp_val;
    });

    return interp_sigt;
  }

  // -----------------------------------------------------------

  Array<Vec<Float, 3>> mesh_vertices(const UniformTopology & mesh);

  Array<Float> side_face_currents(const UniformTopology &mesh,
                                  const UniformIndexer::Side side,
                                  const Array<Float> &vert_sigt,
                                  const Vec<Float, 3> &source,
                                  const Float strength);

  Array<Float> all_face_currents(const UniformTopology & mesh,
                                 const Array<Float> &vert_sigt,
                                 const Vec<Float, 3> &source_,
                                 const Float strength_);

  Array<Float> flux_from_current(const UniformTopology & mesh,
                                 const Vec<Float, 3> &source,
                                 const Float strength,
                                 const Array<Float> &cell_sigt,
                                 const Array<Float> &face_current);

  Float sum_side_current(const UniformTopology &mesh,
                         const UniformIndexer::Side side,
                         const Array<Float> &all_face_current,
                         const Vec<Float, 3> &source);  // orient faces

  Float sum_removal(const UniformTopology &mesh,
                    const Array<Float> &cell_sigt,
                    const Array<Float> &cell_flux);

  // -----------------------------------------------------------

  std::vector<Array<int32>> partition_by_plane(
      const UniformTopology *mesh,
      const std::vector<bool> &planes_viable_,
      const Array<Vec<Float, 3>> world_pts)
  {
    // Return a vector of 6 Arrays.
    // Points can be incident on one, none, or many planes.
    // Don't insert the points in the interior of the domain.
    // The rest of the points, insert into exactly one vector.
    std::vector<Array<int32>> orig_idx_per_plane(6);

    bool planes_viable[6];
    std::copy_n(planes_viable_.begin(), 6, planes_viable);

    const Vec<Float, 3> origin = mesh->origin();
    const Vec<int32, 3> cell_dims = mesh->cell_dims();
    const Vec<Float, 3> spacing = mesh->spacing();
    const Vec<Float, 3> diag = {{cell_dims[0] * spacing[0],
                                 cell_dims[1] * spacing[1],
                                 cell_dims[2] * spacing[2]}};

    const auto almost_equal =
      [=] DRAY_LAMBDA(const Float a, const Float b)
    {
      return fabs(a - b) < epsilon<Float>();
    };

    const auto assign_to_plane =
      [=] DRAY_LAMBDA (const Vec<Float, 3> &pt)
    {
      const Vec<Float, 3> rel = pt - origin;
      using Side = StructuredDomainPanel::Side;

      if (planes_viable[Side::X0] && almost_equal(rel[0], 0))
        return int32(Side::X0);
      if (planes_viable[Side::X1] && almost_equal(rel[0], diag[0]))
        return int32(Side::X1);

      if (planes_viable[Side::Y0] && almost_equal(rel[1], 0))
        return int32(Side::Y0);
      if (planes_viable[Side::Y1] && almost_equal(rel[1], diag[1]))
        return int32(Side::Y1);

      if (planes_viable[Side::Z0] && almost_equal(rel[2], 0))
        return int32(Side::Z0);
      if (planes_viable[Side::Z1] && almost_equal(rel[2], diag[2]))
        return int32(Side::Z1);

      return -1;
    };

    Array<int32> assignments = array_map(world_pts, assign_to_plane);

    for (int32 plane = 0; plane < 6; ++plane)
      orig_idx_per_plane[plane] = index_where(assignments, plane);

    return orig_idx_per_plane;
  }


  // sanity check
  void check_uniform_sigt(const Array<Float> &sigt, const Float expected);

  // sanity check
  void check_uniform_sigt(const Array<Float> &sigt, const Float expected)
  {
    // if uniform absorption case, should be exactly correct; check it here
    const Float diff = array_max_diff(sigt, expected);
    printf("    store diff == %.2e\n", diff);
    if (diff > 1e-6)
    {
      fprintf(stderr, "UNIFORM IS NOT UNIFORM\n");
      exit(1);
    }
  }

}//dray










//
// main()
//
int main (int argc, char *argv[])
{
  init_furnace();

  std::string config_file = "";
  std::string output_file = "egg_carton";

  if (argc != 2)
  {
    std::cout << "Missing configure file name\n";
    exit (1);
  }

  config_file = argv[1];

  Config config (config_file);
  /// config.load_data ();
  /// config.load_camera ();
  /// config.load_field ();
  /// dray::Collection collection = config.m_collection;

  using dray::Vec;
  using dray::Float;
  using dray::int32;

  // sigmat
  double sigmat_amplitude = 1;
  double sigmat_period = 1;
  if (config.m_config.has_child("sigmat_amplitude"))
    sigmat_amplitude = config.m_config["sigmat_amplitude"].to_double();
  if (config.m_config.has_child("sigmat_period"))
    sigmat_period = config.m_config["sigmat_period"].to_double();
  printf("sigmat_amplitude:%f\nsigmat_period:%f\n",
      sigmat_amplitude, sigmat_period);

  // source
  Vec<Float, 3> source = {{0.1, 0.1, 0.1}};
  Float strength = 1.0;

  // dataset mesh
  Vec<Float, 3> global_origin = {{0, 0, 0}};
  Vec<Float, 3> spacing = {{1./64, 1./64, 1./64}};
  /// Vec<int32, 3> domains = {{4, 4, 4}};  //TODO use this, the original test
  Vec<int32, 3> domains = {{1, 1, 2}};  // simpler for debugging
  Vec<int32, 3> cell_dims = {{16, 16, 16}};

  const std::string field_name = "sigt";

  dray::Collection collection;

  const bool use_eggs = true;

  if (use_eggs)
    collection = egg_cartons(
        field_name,
        sigmat_amplitude,
        sigmat_period,
        global_origin,
        spacing,
        domains,
        cell_dims);
  else
    collection = uniform_absorption(
        field_name,
        sigmat_amplitude,
        global_origin,
        spacing,
        domains,
        cell_dims);

  // Output to blueprint for visit.
  conduit::Node conduit_collection;
  for (int dom = 0; dom < collection.local_size(); ++dom)
  {
    conduit::Node & conduit_domain = conduit_collection.append();
    dray::DataSet domain = collection.domain(dom);
    domain.to_blueprint(conduit_domain);
    conduit_domain["state/domain_id"] = domain.domain_id();
  }
  conduit::relay::io::blueprint::save_mesh(
      conduit_collection, output_file + ".blueprint_root_hdf5");

  {
    using namespace dray;
    typedef StructuredDomainPanel SDP;

    // Create panel_list.
    PanelList panel_list;
    const size_t num_domains = collection.local_size();
    std::vector<int32> sockets_per_domain(num_domains);
    size_t domain_i = 0;
    for (DataSet &domain : collection.domains())
    {
      UniformTopology * domain_mesh = structured( domain.mesh() );
      int32 num_sockets = PanelList::num_sockets(domain_mesh);
      DomainToken domain_token = DomainToken::local_domain(domain_i);
      for (int32 socket = 0; socket < num_sockets; ++socket)
      {
        DomainSocket domain_socket = {domain_token, socket};
        panel_list.insert(SDP(domain_mesh, domain_i, socket));
      }
      sockets_per_domain[domain_i] = num_sockets;
      domain_i++;
    }

    // Associate coinciding domain boundary panels.
    using DomainIdx = int32;
    portgraph::PortGraph<DomainIdx> domain_graph;
    for (const std::pair<int32, int32> &ij : panel_list.adjacent_panels())
    {
      const int32 i = ij.first,  j = ij.second;
      const SDP & pi = panel_list[i],  & pj = panel_list[j];

      using namespace portgraph;
      domain_graph.insert(node_port(pi.domain_idx(), pi.socket()),
                          node_port(pj.domain_idx(), pj.socket()));

      // Assumes that each panel has at most one neighbor.
    }

    // for each source... (upwind/downwind depends on source)

    const auto a_less_than_b =
        [=, &panel_list, &domain_graph]( int32 dom_a, int32 dom_b )
    {
      const int32 port_a = domain_graph.from_node(dom_a).to(dom_b).from.port;
      const int32 port_b = domain_graph.from_node(dom_b).to(dom_a).from.port;
      const SDP & panel_a = panel_list(dom_a, port_a);
      const SDP & panel_b = panel_list(dom_b, port_b);
      return a_upwind_of_b(panel_a, panel_b, source);
    };
    std::vector<int32> ordering = topological_order(
        domain_graph, a_less_than_b);


    struct DomainTemporary
    {
      struct Socket
      {
        // Depending on whether the socket is upwind or downwind
        // of the domain, some arrays will be full and others empty.

        // Interpolation surface.
        FaceCurrents m_face_currents;

        // Vertex arrays.
        Array<Vec<Float, 3>> m_out_panel_sample;  // vertices on outflow
        Array<Vec<Float, 3>> m_remainder_entry;   // out-verts -> in-sides
        Array<Float> m_partials;         // computed by domain
        Array<Float> m_out_avg_sigma_t;  // computed by domain
        Array<Float> m_in_avg_sigma_t;   // imported from neighbor

        // Face arrays.
        Array<Float> m_out_current;  // computed by domain
        Array<Float> m_in_current;   // imported from neighbor

        // convenience
        UniformIndexer::Side uiside() const {
          return m_face_currents.m_panel.uiside();
        }
      };

      std::vector<Socket> m_sockets;

      // Face arrays.
      Array<Float> m_all_currents;  // Convention: choose normal for dot > 0

      // Cell arrays.
      Array<Float> m_cell_flux;

      // -----

      void num_sockets(int32 num_sockets) { m_sockets.resize(num_sockets); }
      int32 num_sockets() const { return m_sockets.size(); }

      Socket & socket(int32 socket) { return m_sockets[socket]; }
    };
    std::vector<DomainTemporary> domain_stores(num_domains);

    // Independent work.
    for (int32 domain_idx : ordering)
    {
      DataSet domain = collection.domain(domain_idx);
      DomainTemporary & domain_store = domain_stores[domain_idx];

      domain_store.num_sockets(sockets_per_domain[domain_idx]);
      const UniformTopology * mesh = structured(domain.mesh());
      const LowOrderField * sigt = structured(domain.field(field_name));

      for (int32 socket_id = 0; socket_id < sockets_per_domain[domain_idx]; ++socket_id)
      {
        const SDP & panel = panel_list(domain_idx, socket_id);
        if (panel.downwind_of_domain(source))
        {
          DomainTemporary::Socket & socket = domain_store.socket(socket_id);
          socket.m_face_currents.discretize(mesh, panel, source);
          socket.m_out_panel_sample = socket.m_face_currents.sample_points();
          std::tie( socket.m_partials,
                    socket.m_remainder_entry)
              = uniform_partials(mesh, sigt, source, socket.m_out_panel_sample);
        }
      }
    }

    // 2021-09-21 M.Ishii
    // In future, want to overlap communication and computation.
    // We don't need all dependencies to complete before we can do something.
    // Want to redesign: Push a completed boundary to the next dependent;
    // Once a domain is triggered by the last dependency, does its work
    // and pushes its boundaries; when propagation gets stuck, then
    // wait on the next receive, and hopefully we have more work to do by then.
    //
    // if (there there are domains not dependent on upwind ranks)
    //   push the domain boundaries through the dependency graph,
    //     sending downwind boundaries if possible
    // while (some upwind ranks not received)
    // {
    //   block for the next receive from an upwind rank
    //   push the new domain boundaries through the dependency graph,
    //     sending downwind boundaries if possible
    // }

    // Dependent on upwind neighboring domains (socket.m_in_avg_sigma_t).
    //   Due to regularly-shaped domains, topological ordering of domains
    //   ensures that every panel has its dependencies before interpolating.
    for (int32 domain_idx : ordering)
    {
      DataSet domain = collection.domain(domain_idx);
      const UniformTopology * mesh = structured(domain.mesh());
      const int32 num_sockets = sockets_per_domain[domain_idx];

      const auto partition = [=](
          const Array<Vec<Float, 3>> world_pts,
          const std::vector<bool> &planes_viable)
      {
        return partition_by_plane(mesh, planes_viable, world_pts);
      };

      DomainTemporary & domain_store = domain_stores[domain_idx];

      std::vector<bool> planes_viable(6, false);
      for (int32 socket_id = 0; socket_id < num_sockets; ++socket_id)
      {
        planes_viable[socket_id] =
            panel_list(domain_idx, socket_id).upwind_of_domain(source) &&
            domain_graph.from_node(domain_idx).has_port(socket_id);
      }

      for (int32 socket_id = 0; socket_id < num_sockets; ++socket_id)
      {
        const SDP & panel = panel_list(domain_idx, socket_id);
        if (panel.downwind_of_domain(source))
        {
          DomainTemporary::Socket & socket = domain_store.socket(socket_id);
          const Array<Vec<Float, 3>> remainder_entry = socket.m_remainder_entry;

          Array<Float> interpolated_sigt =
              array_zero<Float>(remainder_entry.size());

          // Separate the interpolation queries by plane, to interpolate easier.
          //   partition() should return same number of Arrays
          //   as there are sockets (6)
          int32 interp_socket_id = 0;
          for (const Array<int32> orig_idx
              : partition(remainder_entry, planes_viable))
          {
            const Array<Vec<Float, 3>> entry_pts = gather(remainder_entry, orig_idx);
            const DomainTemporary::Socket & interp_socket =
                domain_store.socket(interp_socket_id);

            if (entry_pts.size() > 0)
            {
              if (interp_socket.m_in_avg_sigma_t.size() == 0)
                throw std::logic_error("sigt field empty when queried");

              // Interpolate
              const Array<Float> remainder =
                  interp_socket.m_face_currents.interpolate(
                      interp_socket.m_in_avg_sigma_t, entry_pts);

              // Overwrite with per-plane interpolated <Sigma_t>
              scatter(remainder, orig_idx, interpolated_sigt);
            }

            interp_socket_id++;
          }


          // <Sigma_t> between source and panel:
          //   Combine interpolated values and domain-traced values.
          Array<Float> composite_sigt =
              compose_domain_segment( source,
                                      remainder_entry,
                                      interpolated_sigt,
                                      socket.m_out_panel_sample,
                                      socket.m_partials);

          /////////////////
          if (!use_eggs)
            check_uniform_sigt(composite_sigt, sigmat_amplitude);
          /////////////////

          // Store on own outflow boundary panel.
          socket.m_out_avg_sigma_t = composite_sigt;

          // Store on neighboring domain inflow boundary panel.
          if (domain_graph.from_node(domain_idx).has_port(socket_id))
          {
            const portgraph::Link<int32> link =
                domain_graph.from_node(domain_idx).port(socket_id);

            DomainTemporary::Socket & nbr_socket =
                domain_stores[link.to.node].socket(link.to.port);

            nbr_socket.m_in_avg_sigma_t = composite_sigt;
          }
        }
      }
    }

    // After outflow sigt is sent downwind,
    // compute outflow *face currents* and send downwind
    // (owned by upwind for consistency).

    for (int32 domain_idx : ordering)
    {
      DataSet domain = collection.domain(domain_idx);
      const UniformTopology * mesh = structured(domain.mesh());
      const int32 num_sockets = sockets_per_domain[domain_idx];

      DomainTemporary & domain_store = domain_stores[domain_idx];

      for (int32 socket_id = 0; socket_id < num_sockets; ++socket_id)
      {
        const SDP & panel = panel_list(domain_idx, socket_id);
        if (panel.downwind_of_domain(source))
        {
          DomainTemporary::Socket & socket = domain_store.socket(socket_id);
          const Array<Float> vert_sigt = socket.m_out_avg_sigma_t;

          const Array<Float> current = side_face_currents(
              *mesh, panel.uiside(), vert_sigt, source, strength);

          // Store to our outflow boundary
          socket.m_out_current = current;

          // Store to neighbor inflow boundary
          if (domain_graph.from_node(domain_idx).has_port(socket_id))
          {
            const portgraph::Link<int32> link =
                domain_graph.from_node(domain_idx).port(socket_id);

            DomainTemporary::Socket & nbr_socket =
                domain_stores[link.to.node].socket(link.to.port);

            nbr_socket.m_in_current = current;
          }
        }
      }
    }
    // future: 2nd communication over mpi
    //  No long dependency chains here, just one link long.


    // Average sigt is at the inflow boundaries.
    // Upwind face currents are also at the inflow boundaries.
    // To get face currents for all other faces,
    //   1. Trace sigt to all vertices (non-inflow)
    //   2. Integrate current moments on all faces (non-inflow),
    //      using interpolated sigt.
    // Overwrite inflow face currents with the received values.

    for (int32 domain_idx : ordering)
    {
      DataSet domain = collection.domain(domain_idx);
      const UniformTopology * mesh = structured(domain.mesh());
      const LowOrderField * sigt = structured(domain.field(field_name));
      const int32 num_sockets = sockets_per_domain[domain_idx];
      const UniformIndexer idxr = {mesh->cell_dims()};

      DomainTemporary & domain_store = domain_stores[domain_idx];
      using Socket = DomainTemporary::Socket;

      // trace to vertices:  vertices -> (partials, intercepts)
      const Array<Vec<Float, 3>> vertices = mesh_vertices(*mesh);
      Array<Float> partials;
      Array<Vec<Float, 3>> intercepts;
      std::tie( partials, intercepts )
          = uniform_partials(mesh, sigt, source, vertices);

      // composite:  -> all_verts_sigt
      Array<Float> all_verts_sigt;
      {
        // Interpolate sigt from inflow boundary on intercepted faces.
        Array<Float> intercept_sigt = array_zero<Float>(intercepts.size());

        // duplicate code for planes_viable
        std::vector<bool> planes_viable(6, false);
        for (int32 socket_id = 0; socket_id < num_sockets; ++socket_id)
        {
          planes_viable[socket_id] =
              panel_list(domain_idx, socket_id).upwind_of_domain(source) &&
              domain_graph.from_node(domain_idx).has_port(socket_id);
        }

        int32 intercept_id = 0;
        for (const Array<int32> orig_idx
            : partition_by_plane(mesh, planes_viable, intercepts))
        {
          const Array<Vec<Float, 3>> entry_pts = gather(intercepts, orig_idx);
          const Socket & intercept_socket = domain_store.socket(intercept_id);
          const Array<Float> &in_sigt = intercept_socket.m_in_avg_sigma_t;
          const FaceCurrents & interp_surf = intercept_socket.m_face_currents;

          if (entry_pts.size() > 0)
          {
            if (in_sigt.size() == 0)
              throw std::logic_error("sigt field empty when queried");
            const Array<Float> prefix =
                interp_surf.interpolate(in_sigt, entry_pts);
            scatter(prefix, orig_idx, intercept_sigt);
          }
          intercept_id++;
        }

        // <Sigma_t> between source and vertices:
        //   Combine interpolated values and domain-traced values.
        all_verts_sigt = compose_domain_segment(
            source, intercepts, intercept_sigt, vertices, partials);
      }

      // overwrite inflow boundary vertex sigt from import
      for (int32 side = 0; side < num_sockets; ++side)
      {
        const Socket & socket = domain_store.socket(side);
        if (socket.m_in_avg_sigma_t.size() > 0)
        {
          idxr.scatter( idxr.side_vert_set(socket.uiside()),
                        socket.m_in_avg_sigma_t,
                        all_verts_sigt);
        }
      }

      // Sanity check:
      // Compare values on the outflow boundaries to those
      // previously computed (prior to exporting sigt).
      for (int32 iside = 0; iside < num_sockets; ++iside)
      {
        const Socket & socket = domain_store.socket(iside);
        const UniformIndexer::Side side = socket.uiside();
        if (socket.m_out_avg_sigma_t.size() > 0)
        {
          UniformIndexer::SideVertSet side_set = idxr.side_vert_set(side);
          Array<Float> fresh_sigt = array_zero<Float>(idxr.side_verts_size(side_set));
          idxr.gather(side_set, fresh_sigt, all_verts_sigt);
          const Float diff = array_max_diff(fresh_sigt, socket.m_out_avg_sigma_t);
          fprintf(stderr, "\t\t\t\tdiff(fresh, out_sigt)==%.5f\n", diff);
          //TODO figure out why they are so different?
        }
      }


      // integrate all face currents
      Array<Float> all_currents = all_face_currents(
          *mesh, all_verts_sigt, source, strength);

      // Sanity check:
      // Compare values on the outflow boundaries to those
      // previously computed (prior to exporting current).
      for (int32 iside = 0; iside < num_sockets; ++iside)
      {
        const Socket & socket = domain_store.socket(iside);
        const UniformIndexer::Side side = socket.uiside();
        if (socket.m_out_current.size() > 0)
        {
          UniformIndexer::SideFaceSet side_set = idxr.side_face_set(side);
          Array<Float> fresh_current = array_zero<Float>(idxr.side_faces_size(side_set));
          idxr.gather(side_set, fresh_current, all_currents);
          const Float diff = array_max_diff(fresh_current, socket.m_out_current);
          fprintf(stderr, "diff(fresh, out_current)==%.10f\n", diff);
          // TODO check gather/scatter compatibility --> t_dray_uniform_indexer.cpp
        }
      }

      // overwrite inflow boundary face currents from import
      for (int32 side = 0; side < num_sockets; ++side)
      {
        const Socket & socket = domain_store.socket(side);
        if (socket.m_in_current.size() > 0)
        {
          idxr.scatter( idxr.side_face_set(socket.uiside()),
                        socket.m_in_current,
                        all_currents );
        }
      }

      domain_store.m_all_currents = all_currents;

      // Note that for debugging and visualization it might also be
      // handy to divide by face areas and store face-averaged flux.
    }

    // Currents (future: moments) are on all faces.
    // Compute DivThm-based cell-centered flux on all cells.
    //
    // This is inside the loop-over-sources, because inflow/outflow
    // of faces depends on source position relative to the cell,
    // and each cell must figure out the orientations of its faces.
    for (int32 domain_idx : ordering)
    {
      DataSet domain = collection.domain(domain_idx);
      const UniformTopology * mesh = structured(domain.mesh());
      LowOrderField * sigt = structured(domain.field(field_name));
      const int32 num_sockets = sockets_per_domain[domain_idx];
      const UniformIndexer idxr = {mesh->cell_dims()};

      DomainTemporary & domain_store = domain_stores[domain_idx];
      domain_store.m_cell_flux = flux_from_current(
          *mesh, source, strength, sigt->values(), domain_store.m_all_currents);
    }

    //
    // Check conservation:  Leakage = source - removal
    //

    // Conservation over each domain individually...
    for (int32 domain_idx : ordering)
    {
      using UI = UniformIndexer;
      DataSet domain = collection.domain(domain_idx);
      const UniformTopology * mesh = structured(domain.mesh());
      LowOrderField * sigt = structured(domain.field(field_name));

      Float leakage = 0;
      for (int32 side = 0; side < UI::NUM_SIDES; ++side)
      {
        leakage += sum_side_current(
            *mesh,
            UI::side(side),
            domain_stores[domain_idx].m_all_currents,
            source);
      }

      Float source_integral = 0;
      if (mesh->locate(source).m_cell_id >= 0)
        source_integral = strength;

      Float removal = sum_removal(
          *mesh,
          sigt->values(),
          domain_stores[domain_idx].m_cell_flux);

      const Float excess = leakage - (source_integral - removal);
      fprintf(stdout, "Domain[%2d] excess==% .11f\n", domain_idx, excess);
    }

    // Conservation over the whole problem...
    //   need to loop over domains and sides, but only add current
    //   for the sides that aren't connected to anything.
    {
      double leakage = 0;
      double source_integral = 0;
      double removal = 0;

      double cancelling_current = 0;

      for (int32 domain_idx : ordering)
      {
        DataSet domain = collection.domain(domain_idx);
        const UniformTopology * mesh = structured(domain.mesh());
        LowOrderField * sigt = structured(domain.field(field_name));

        DomainTemporary & domain_store = domain_stores[domain_idx];
        const int32 num_sockets = sockets_per_domain[domain_idx];

        // Leakage
        double domain_leakage = 0;
        for (int32 socket_id = 0; socket_id < num_sockets; ++socket_id)
        {
          const SDP & panel = panel_list(domain_idx, socket_id);
          UniformIndexer::Side side = panel.uiside();

          if (!domain_graph.from_node(domain_idx).has_port(socket_id))
          {
            // Then it's part of the boundary of the whole problem.
            domain_leakage += sum_side_current(
                *mesh,
                side,
                domain_stores[domain_idx].m_all_currents,
                source);
          }
          else
          {
            // Then it should be cancelled by another domain.
            cancelling_current += sum_side_current(
                *mesh,
                side,
                domain_stores[domain_idx].m_all_currents,
                source);
          }
        }
        leakage += domain_leakage;

        // Source
        if (mesh->locate(source).m_cell_id >= 0)
          source_integral = strength;
        // If source is in the problem (at least one domain),
        // then get value of strength, else remains 0.

        // Leakage
        removal += sum_removal(
          *mesh,
          sigt->values(),
          domain_stores[domain_idx].m_cell_flux);
      }

      const Float excess = leakage - (source_integral - removal);
      fprintf(stdout, "Entire     excess==% .11f\n", excess);
      fprintf(stdout, "  leak:%.2e  src:%.2e  rem:%.2e\n",
          leakage, source_integral, removal);
      fprintf(stdout, "  cancelling==% .11f\n", cancelling_current);
      fprintf(stdout, "excess+cancelling==%e\n", excess + cancelling_current);
    }

    // future: consider moments
    /// // Flux moments are on cells.
    /// // Add to total flux moments over all sources.

    // OUTSIDE THE LOOP OVER SOURCES
  }

  {
    using namespace dray;

    // There may be many domains on each of many MPI ranks.
    // Domains downwind (further from the source) depend on
    // those upwind (nearer to the source).
    //
    // Tasks for each domain:
    //   1. I-Send own boundary shape to downwind domains.
    //   2. W-Recv boundary shapes from upwind domains.
    //   3. I-Recv boundary data from upwind domains.
    //   4. Independent ray-trace.
    //   5. W-Recv boundary data from upwind domains.
    //   6. Interpolate and add from upwind to downwind boundaries.
    //   7. Send boundary data to downwind domains.

    // Hybrid of single-threaded DFS and multi-process MPI.
    //
    //   // Exchange boundary shapes, to init data structures.
    //   for (domain)
    //     if (outside depends on domain)
    //       I-Send downwind domain boundary shape
    //   for (domain)
    //     if (domain depends on outside)
    //       W-Recv upwind domain boundary shape
    //     else
    //       Locally ask for upwind domain boundary shape
    //
    //   //
    //   // Interpolation Surfaces task graph.
    //   //
    //
    //   // Post all recvs.
    //   for (domain)
    //     if (domain depends on outside)
    //       Begin I-Recv upwind domain boundary I.S.
    //
    //   // Do independent work.
    //   for (domain)
    //     Trace and store domain partials
    //
    //   // Do dependent work.  (See DFS get_downwind() below)
    //   for (domain)
    //     if (outside depends)
    //       StoreAndSend( prepare_downwind(domain) )
    //   for (domain)
    //     if (outside does not depend)
    //       Store( prepare_downwind(domain) )
    //
    //
    //   WHERE
    //       def prepare_downwind(domain):
    //         if (ready)
    //           return it
    //         else
    //           for each upwind domain'
    //             if (domain' is outside)
    //               End W-Recv upwind domain boundary I.S.
    //             else
    //               Stage prepare_downwind(domain')
    //           Interpolate and add, downwind=owned+upwind
    //           ready=true
    //           return downwind I.S.

    // for each source:
      /// for (DataSet domain : collection.domains())
      /// {
      ///   UniformTopology * mesh = structured(domain.mesh());
      ///   LowOrderField * sigt = structured(domain.field(field_name));

      ///   // Distance-, angle-, and resolution-dependent quadrature rule.
      ///   FaceCurrents face_currents(mesh, source);

      ///   // Where to trace to.
      ///   Array<Vec<Float, 3>> sample_points = face_currents.sample_points();

      ///   // Trace from direction of the source, in this domain.
      ///   Array<Float> in_domain_opt_depth =
      ///       uniform_partials(mesh, sigt, source, sample_points);
      ///   face_currents.store_domain_partials(in_domain_opt_depth);
      /// }

      /// // exchange

      /// for (DataSet domain : collection.domains())
      /// {

      /// }
  }



  //TODO 1.   Specify locations of interpolation surfaces
  //TODO 2.   Modify/duplicate the first_scatter filter to accept multidomains
  //            and interpolation surfaces
  //               -- Compute/estimate <sigmat> at vertices
  //               -- Compute currents
  //               -- Calculate per-cell <psi>
  //               : should be able to match leakage and removal rate, should match by construction
  //TODO 3.   Compute the leakage
  //TODO 4.   Compute the positivity

  finalize_furnace();
}


static dray::Collection regular_domain_collection(
    const dray::Vec<dray::Float, 3> &origin,
    const dray::Vec<dray::Float, 3> &spacing,
    const dray::Vec<dray::int32, 3> &domains_layout,
    const dray::Vec<dray::int32, 3> &cells_per_domain)
{
  using namespace dray;
  Collection collection;

  // Mesh.
  int32 domain_id = 0;
  int32 domain_i[3];
  int32 global_cell[3];
  for (domain_i[2] = 0; domain_i[2] < domains_layout[2]; ++domain_i[2])
  {
    global_cell[2] = domain_i[2] * cells_per_domain[2];
    for (domain_i[1] = 0; domain_i[1] < domains_layout[1]; ++domain_i[1])
    {
      global_cell[1] = domain_i[1] * cells_per_domain[1];
      for (domain_i[0] = 0; domain_i[0] < domains_layout[0]; ++domain_i[0])
      {
        global_cell[0] = domain_i[0] * cells_per_domain[0];

        // domain_origin
        Vec<Float, 3> domain_origin = origin;
        for (int d = 0; d < 3; ++d)
          domain_origin[d] += spacing[d] * global_cell[d];

        // Initialize mesh.
        // domain spacing and cell dims are "spacing" and "cells_per_domain"
        std::shared_ptr<UniformTopology> mesh =
            std::make_shared<UniformTopology>(
                spacing, domain_origin, cells_per_domain);
        mesh->name("topo");

        // Add domain to collection.
        DataSet domain(mesh);
        domain.domain_id(domain_id);
        collection.add_domain(domain);

        domain_id++;
      }
    }
  }

  return collection;
}



dray::Collection egg_cartons(
    const std::string &field_name,
    dray::float64 sigmat_amplitude,
    dray::float64 sigmat_period,
    const dray::Vec<dray::Float, 3> &origin,
    const dray::Vec<dray::Float, 3> &spacing,
    const dray::Vec<dray::int32, 3> &domains_layout,
    const dray::Vec<dray::int32, 3> &cells_per_domain)
{
  using namespace dray;
  Collection collection = regular_domain_collection(
      origin, spacing, domains_layout, cells_per_domain);

  // Add egg carton field.
  const auto sigmat_cell_avg = [=] (
      const Vec<Float, 3> &lo,
      const Vec<Float, 3> &hi,
      float64 amplitude,
      float64 period)
  {
    float64 product = 1;
    for (int d = 0; d < 3; ++d)
    {
      // Definite integral avg of sin^2(pi * x[d] / p) / (width[d])
      float64 factor =
          (0.5
          + period / (4 * pi() * (hi[d] - lo[d]))
            * (sin(lo[d] * 2 * pi() / period) - sin(hi[d] * 2 * pi() / period)));
      product *= factor;
    }
    return amplitude * product;
  };

  const auto add_field_to_domain = [=, &sigmat_cell_avg](DataSet &domain)
  {
    UniformTopology * mesh = dynamic_cast<UniformTopology *>(domain.mesh());
    const Vec<int32, 3> dims = mesh->cell_dims();
    const Vec<Float, 3> spacing = mesh->spacing();
    const Vec<Float, 3> origin = mesh->origin();

    int32 size = dims[0] * dims[1] * dims[2];

    Array<Float> sigt;
    sigt.resize(size);
    Float * sigt_ptr = sigt.get_host_ptr();

    int32 i[3];
    int32 global_cell[3];
    for (i[2] = 0; i[2] < dims[2]; ++i[2])
      for (i[1] = 0; i[1] < dims[1]; ++i[1])
        for (i[0] = 0; i[0] < dims[0]; ++i[0])
        {
          const int32 offset = i[0] + dims[0] * (i[1] + dims[1] * i[2]);
          Vec<Float, 3> xmin = {{i[0] * spacing[0],
                                 i[1] * spacing[1],
                                 i[2] * spacing[2]}};
          xmin += origin;
          const Vec<Float, 3> xmax = xmin + spacing;

          Float value = sigmat_cell_avg(
              xmin, xmax, sigmat_amplitude, sigmat_period);
          sigt_ptr[offset] = value;
        }

    std::shared_ptr<LowOrderField> field =
        std::make_shared<LowOrderField>(
            sigt, LowOrderField::Assoc::Element, dims);
    field->name(field_name);
    domain.add_field(field);
  };

  for (DataSet &domain : collection.domains())
    add_field_to_domain(domain);

  return collection;
}


dray::Collection uniform_absorption(
    const std::string &field_name,
    dray::float64 absorption,
    const dray::Vec<dray::Float, 3> &origin,
    const dray::Vec<dray::Float, 3> &spacing,
    const dray::Vec<dray::int32, 3> &domains_layout,
    const dray::Vec<dray::int32, 3> &cells_per_domain)
{
  using namespace dray;
  Collection collection = regular_domain_collection(
      origin, spacing, domains_layout, cells_per_domain);

  const auto add_field_to_domain = [=](DataSet &domain)
  {
    UniformTopology * mesh = dynamic_cast<UniformTopology *>(domain.mesh());
    const Vec<int32, 3> dims = mesh->cell_dims();
    const Vec<Float, 3> spacing = mesh->spacing();
    const Vec<Float, 3> origin = mesh->origin();

    int32 size = dims[0] * dims[1] * dims[2];

    Array<Float> sigt;
    sigt.resize(size);
    Float * sigt_ptr = sigt.get_host_ptr();

    int32 i[3];
    for (i[2] = 0; i[2] < dims[2]; ++i[2])
      for (i[1] = 0; i[1] < dims[1]; ++i[1])
        for (i[0] = 0; i[0] < dims[0]; ++i[0])
        {
          const int32 offset = i[0] + dims[0] * (i[1] + dims[1] * i[2]);
          Float value = absorption;
          sigt_ptr[offset] = value;
        }

    std::shared_ptr<LowOrderField> field =
        std::make_shared<LowOrderField>(
            sigt, LowOrderField::Assoc::Element, dims);
    field->name(field_name);
    domain.add_field(field);
  };

  for (DataSet &domain : collection.domains())
    add_field_to_domain(domain);

  return collection;
}



namespace dray
{

  DRAY_EXEC Float face_current(
      const int32 subdiv,   // number of points in each axis (tensor)
      const Vec<Float, 3> &source,
      const Float strength,
      const Vec<Float, 3> verts[4],
      const Float sigt[4],
      const Float area,
      const Vec<Float, 3> &normal)
  {
    double sum = 0;

    // Integrate{ q / r^2 * exp(-<sigt>*r) * dot(r,n) dA }

    // Open uniform quadrature with linear interpolation of Sigma_t.
    int32 i[2];
    Float t[2];
    Float w[2];
    const int32 n = subdiv + 1;
    for (i[1] = 0; i[1] < subdiv; ++i[1])
    {
      t[1] = (i[1] + 1) * 1.0 / n;
      w[1] = 1.0 / subdiv;

      const Vec<Float, 3> V0 = lerp(verts[0], verts[2], t[1]);
      const Vec<Float, 3> V1 = lerp(verts[1], verts[3], t[1]);
      const Float S0 = lerp(sigt[0], sigt[2], t[1]);
      const Float S1 = lerp(sigt[1], sigt[3], t[1]);

      for (i[0] = 0; i[0] < subdiv; ++i[0])
      {
        t[0] = (i[0] + 1) * 1.0 / n;
        w[0] = 1.0 / subdiv;

        const Vec<Float, 3> v = lerp(V0, V1, t[0]);
        const Float s = lerp(S0, S1, t[0]);
        const Float weight = w[0] * w[1];

        const Vec<Float, 3> r = v - source;
        const Vec<Float, 3> r_hat = r.normalized();
        const Float r_mag2 = r.magnitude2();
        const Float r_mag = sqrt(r_mag2);

        sum += exp(-s * r_mag) * rcp_safe(r_mag2) * dot(r_hat, normal) * weight;
      }
    }

    return sum * strength * area;
  }



  Array<Float> side_face_currents(const UniformTopology & mesh,
                                  const UniformIndexer::Side side,
                                  const Array<Float> &vert_sigt,
                                  const Vec<Float, 3> &source_,
                                  const Float strength_)
  {
    const UniformIndexer idxr = {mesh.cell_dims()};
    const Vec<Float, 3> origin = mesh.origin();
    const Vec<Float, 3> spacing = mesh.spacing();
    const UniformIndexer::SideFaceSet side_set = {side};
    const size_t size = idxr.side_faces_size(side_set);

    Array<Float> current = array_zero<Float>(size);
    NonConstDeviceArray<Float> d_current(current);
    ConstDeviceArray<Float> d_sigt(vert_sigt);

    const Vec<int32, 3> logical_normal = UniformIndexer::normal(side);
    Vec<Float, 3> unit_spacing = {{1, 1, 1}};
    for (int32 d = 0; d < 3; ++d)
      if (spacing[d] < 0)
        unit_spacing[d] = -1;

    const Vec<Float, 3> normal = hadamard(logical_normal.to<Float>(), unit_spacing);
    const Vec<Float, 2> edges = sub_vec<2>(spacing, idxr.axis_subset(side));
    const Float area = edges[0] * edges[1];

    const Vec<Float, 3> source = source_;  // avoid lambda capture issues
    const Float strength = strength_;

    RAJA::forall<for_policy>(RAJA::RangeSegment(0, size),
        [=] DRAY_LAMBDA (int32 i)
    {
      const UniformIndexer::SideFaces side_faces = idxr.side_faces(side, i);

      // Vertices.
      Vec<Float, 3> verts[4];
      Float sigt[4];
      for (int32 corner = 0; corner < 4; ++corner)
      {
        const UniformIndexer::SideVerts side_verts =
            idxr.side_verts(side_faces, corner);
        const Vec<Float, 3> vert_idx_float = side_verts.idx.to<Float>();
        verts[corner] = origin + hadamard(spacing, vert_idx_float);
        sigt[corner] = d_sigt.get_item(idxr.flat_idx(side_verts));
      }

      // Integrate current numerically.
      const int32 subdiv = 3;
      const Float current = face_current(subdiv, source, strength, verts, sigt, area, normal);

      d_current.get_item(i) = current;
    });

    return current;
  }


  Array<Float> all_face_currents(const UniformTopology & mesh,
                                 const Array<Float> &vert_sigt,
                                 const Vec<Float, 3> &source_,
                                 const Float strength_)
  {
    const UniformIndexer idxr = {mesh.cell_dims()};
    const Vec<Float, 3> origin = mesh.origin();
    const Vec<Float, 3> spacing = mesh.spacing();
    const size_t size = idxr.all_faces_size();

    Array<Float> current = array_zero<Float>(size);
    NonConstDeviceArray<Float> d_current(current);
    ConstDeviceArray<Float> d_sigt(vert_sigt);

    const Vec<Float, 3> source = source_;  // avoid lambda capture issues
    const Float strength = strength_;

    RAJA::forall<for_policy>(RAJA::RangeSegment(0, size),
        [=] DRAY_LAMBDA (int32 i)
    {
      const UniformIndexer::AllFaces all_faces = idxr.all_faces(i);
      const UniformIndexer::Plane plane = all_faces.plane;
      Vec<Float, 3> normal = idxr.normal(plane).to<Float>();
      const Vec<Float, 2> edges = sub_vec<2>(spacing, idxr.axis_subset(plane));
      const Float area = edges[0] * edges[1];

      // Vertices.
      Vec<Float, 3> verts[4];
      Float sigt[4];
      Float max_abs_dot = 0;
      for (int32 corner = 0; corner < 4; ++corner)
      {
        const UniformIndexer::AllVerts all_verts =
            idxr.all_verts(all_faces, corner);
        const Vec<Float, 3> vert_idx_float = all_verts.idx.to<Float>();
        verts[corner] = origin + hadamard(spacing, vert_idx_float);
        sigt[corner] = d_sigt.get_item(idxr.flat_idx(all_verts));

        const Float new_dot = dot(normal, (verts[corner] - source));
        if (abs(new_dot) > abs(max_abs_dot))
          max_abs_dot = new_dot;
      }

      if (max_abs_dot < 0)
        normal = -normal;

      // Integrate current numerically.
      const int32 subdiv = 3;
      const Float current = face_current(subdiv, source, strength, verts, sigt, area, normal);

      d_current.get_item(i) = current;
    });

    return current;
  }


  Array<Float> flux_from_current(const UniformTopology & mesh,
                                 const Vec<Float, 3> &source_,
                                 const Float strength,
                                 const Array<Float> &cell_sigt,
                                 const Array<Float> &face_current)
  {
    const UniformIndexer idxr = {mesh.cell_dims()};
    const Vec<Float, 3> origin = mesh.origin();
    const Vec<Float, 3> spacing = mesh.spacing();
    const Float cell_volume = spacing[0] * spacing[1] * spacing[2];
    const size_t size = idxr.all_cells_size();
    using UI = UniformIndexer;

    Array<Float> flux;
    flux.resize(size);
    NonConstDeviceArray<Float> d_flux(flux);
    ConstDeviceArray<Float> d_cell_sigt(cell_sigt);
    ConstDeviceArray<Float> d_face_current(face_current);

    const Vec<Float, 3> source = source_;  // avoid lambda capture issues

    RAJA::forall<for_policy>(RAJA::RangeSegment(0, size),
        [=] DRAY_LAMBDA (int32 i)
    {
      const UI::AllCells cell = idxr.all_cells(i);

      // Sum the face currents.
      double sum_current = 0;
      for (int32 iside = 0; iside < UI::NUM_SIDES; ++iside)
      {
        const UI::Side side = UI::side(iside);
        const UI::AllFaces face = idxr.all_faces(cell, side);
        const UI::AllVerts some_vertex = idxr.all_verts(face, 0);

        // Currents are stored in positive orientation.
        // Might need to flip it depending which side of the face
        // this cell is on.
        const Vec<int32, 3> logical_normal = UI::normal(side);
        Vec<Float, 3> normal = logical_normal.to<Float>();
        for (int32 d = 0; d < 3; ++d)
          if (spacing[d] < 0)
            normal[d] = -normal[d];
        const Vec<Float, 3> vert_idx_float = some_vertex.idx.to<Float>();
        const Vec<Float, 3> vert_pt = origin + hadamard(spacing, vert_idx_float);
        const bool dot_negative = (dot(normal, (vert_pt - source)) < 0);

        Float current = d_face_current.get_item(idxr.flat_idx(face));
        if (dot_negative)
          current = -current;

        sum_current += current;
      }

      // Get the cell-averaged sigt.
      const Float sigt = d_cell_sigt.get_item(i);

      // Apply the divergence theorem, approximating with cell-avg values.
      const Float flux = -sum_current / (sigt * cell_volume);

      d_flux.get_item(i) = flux;
    });

    // If one of the cells has source in it, that also adds to the flux.
    // `strength' for a point source is the volume integral (of a delta),
    // so flux = (strength - sum_currents) / (sigt * volume)
    Location source_location = mesh.locate(source);
    if (source_location.m_cell_id >= 0)
    {
      const int32 source_cell = source_location.m_cell_id;
      const Float sigt = cell_sigt.get_host_ptr_const()[source_cell];
      const Float source_contrib = strength / (sigt * cell_volume);

      flux.get_host_ptr()[source_cell] += source_contrib;
    }

    return flux;
  }


  Float sum_side_current(const UniformTopology &mesh,
                         const UniformIndexer::Side side_,
                         const Array<Float> &all_face_current,
                         const Vec<Float, 3> &source_)
  {
    using UI = UniformIndexer;
    const UniformIndexer idxr = {mesh.cell_dims()};
    const Vec<Float, 3> origin = mesh.origin();
    const Vec<Float, 3> spacing = mesh.spacing();

    const UI::Side side = side_;
    const Vec<Float, 3> source = source_;

    const UI::SideFaceSet side_set = {side};
    const size_t size = idxr.side_faces_size(side_set);

    ConstDeviceArray<Float> d_all_face_current(all_face_current);

    RAJA::ReduceSum<reduce_policy, double> total_current(0);
    RAJA::forall<for_policy>(RAJA::RangeSegment(0, size),
        [=] DRAY_LAMBDA (int32 i)
    {
      // Lookup
      const UI::SideFaces side_faces = idxr.side_faces(side, i);
      const UI::AllFaces all_faces = idxr.all_faces(side_faces);
      Float current = d_all_face_current.get_item(idxr.flat_idx(all_faces));

      // Normal flip
      const Vec<int32, 3> logical_normal = UI::normal(side);
      Vec<Float, 3> normal = logical_normal.to<Float>();
      for (int32 d = 0; d < 3; ++d)
        if (spacing[d] < 0)
          normal[d] = -normal[d];
      const Vec<Float, 3> idx_float = all_faces.idx.to<Float>();
      const Vec<Float, 3> pt = origin + hadamard(spacing, idx_float);
      const bool dot_negative = (dot(normal, (pt - source)) < 0);
      if (dot_negative)
        current = -current;

      total_current += current;
    });

    return total_current.get();
  }


  Float sum_removal(const UniformTopology &mesh,
                    const Array<Float> &cell_sigt,
                    const Array<Float> &cell_flux)
  {
    const UniformIndexer idxr = {mesh.cell_dims()};
    const Vec<Float, 3> spacing = mesh.spacing();
    const Float cell_volume = spacing[0] * spacing[1] * spacing[2];
    const size_t size = idxr.all_cells_size();

    ConstDeviceArray<Float> d_cell_sigt(cell_sigt);
    ConstDeviceArray<Float> d_cell_flux(cell_flux);

    RAJA::ReduceSum<reduce_policy, double> total_removal(0);
    RAJA::forall<for_policy>(RAJA::RangeSegment(0, size),
        [=] DRAY_LAMBDA (int32 i)
    {
      const Float sigt = d_cell_sigt.get_item(i);
      const Float flux = d_cell_flux.get_item(i);
      total_removal += sigt * flux * cell_volume;
    });

    return total_removal.get();
  }



  Array<Vec<Float, 3>> mesh_vertices(const UniformTopology & mesh)
  {
    const UniformIndexer idxr = {mesh.cell_dims()};
    const Vec<Float, 3> origin = mesh.origin();
    const Vec<Float, 3> spacing = mesh.spacing();
    const size_t size = idxr.all_verts_size();

    Array<Vec<Float, 3>> verts;
    verts.resize(size);
    NonConstDeviceArray<Vec<Float, 3>> d_verts(verts);

    RAJA::forall<for_policy>(RAJA::RangeSegment(0, size),
        [=] DRAY_LAMBDA (int32 i)
    {
      const UniformIndexer::AllVerts all_verts = idxr.all_verts(i);
      const Vec<Float, 3> idx_float = all_verts.idx.to<Float>();
      const Vec<Float, 3> pt = origin + hadamard(spacing, idx_float);
      d_verts.get_item(i) = pt;
    });

    return verts;
  }

}


