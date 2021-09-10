// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <dray/dray.hpp>
#include <dray/utils/appstats.hpp>
#include <dray/utils/png_encoder.hpp>

#include <dray/uniform_topology.hpp>
#include <dray/data_model/low_order_field.hpp>
#include <dray/data_model/data_set.hpp>
#include <dray/data_model/collection.hpp>

#include <dray/transport/uniform_partials.hpp>

#include "parsing.hpp"
#include <conduit.hpp>
#include <conduit_blueprint.hpp>
#include <conduit_relay.hpp>
#include <iostream>
#include <random>

//---------- prototype
//----------

dray::Collection egg_cartons(
    const std::string &field_name,
    dray::float64 sigmat_amplitude,
    dray::float64 sigmat_period,
    const dray::Vec<dray::Float, 3> &origin,
    const dray::Vec<dray::Float, 3> &spacing,
    const dray::Vec<dray::int32, 3> &domains_layout,
    const dray::Vec<dray::int32, 3> &cells_per_domain);




namespace dray
{
  namespace adapt
  {
    // Adapter to range-based for-loop syntax.
    template <typename IteratorT>
    struct Range
    {
      IteratorT m_begin;
      IteratorT m_end;

      Range(const std::pair<IteratorT, IteratorT> &begin_end)
        : Range(begin_end.first, begin_end.second)
      {}

      Range(IteratorT begin, IteratorT end)
        : m_begin(begin), m_end(end)
      {}

      Range(const Range &) = default;
      Range(Range &&) = default;

      // Expected by range-based for-loop.
      IteratorT begin() const { return m_begin; }
      IteratorT end() const { return m_end; }
    };
  }//adapt


  /**
   * Digraph: Directed graph.
   *   Operations have complexity O(log(E)).
   */
  class Digraph
  {
    protected:
      std::multimap<int32, int32> m_edges;

    public:
      Digraph();

      void insert_edge(int32 node, int32 neighbor);
      int32 count_edges(int32 node) const;
      /// int32 neighbor(int32 node, int32 edge) const;

      using MapType = std::multimap<int32, int32>;
      using MapIterator = typename MapType::const_iterator;
      using NeighborRange = adapt::Range<MapIterator>;

      NeighborRange neighbors(int32 node) const;
  };

  // -----------------------------
  // Digraph definitions
  // -----------------------------
  Digraph::Digraph()
  {}

  void Digraph::insert_edge(int32 node, int32 neighbor)
  {
    m_edges.insert({node, neighbor});
  }

  int32 Digraph::count_edges(int32 node) const
  {
    return m_edges.count(node);
  }

  Digraph::NeighborRange Digraph::neighbors(int32 node) const
  {
    return NeighborRange(m_edges.equal_range(node));
  }

  /// int32 Digraph::neighbor(int32 node, int32 edge) const
  /// {
  ///   std::multimap<int32, int32>::const_iterator it
  ///       = m_edges.lower_bound(node)
  ///   while (edge-- > 0)
  ///     ++it;
  ///   return it->second;
  /// }
  // -----------------------------
  


  // Precondition: `nearby_domains` contains all domains
  //               adjacent to any in `rank_domains`

  // for each domain in nearby_domains:
  //   produce panels
  //     all_pairs <- Pair(panel, domain, panel is {upwind/downwind} of domain);
  //     all_panels <- panel

  // separate all_panels by plane (orientation and sheet displacement)

  // Digraph upwind;       // (i,j): j is upwind of i
  // Digraph downwind;     // (i,j): j is downwind of i

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

  // knows about StructuredDomains and assigning panels to faces
  class StructuredDomainPanel
  {
    /// protected:
    public:
      int32 m_domain_idx;
      int32 m_socket;
      /// bool m_upwind_of_domain;   // depends on source location
      Vec<Float, 3> m_origin;
      Vec<Float, 3> m_spacing;
      Vec<int32, 3> m_cell_dims;
      Vec<uint8, 3> m_axis_subset;
      enum Side { X0, X1, Y0, Y1, Z0, Z1, NUM_SIDES };
      Side m_side;

      friend class PanelList;

    public:
      StructuredDomainPanel(
          const UniformTopology *domain_mesh,
          int32 domain_idx,
          int32 socket);

      StructuredDomainPanel(const StructuredDomainPanel &) = default;
      StructuredDomainPanel(StructuredDomainPanel &&) = default;

      int32 domain_idx() const;
      int32 socket() const;
      bool upwind_of_domain(const Vec<Float, 3> &source) const;

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

  template <int sub_dim, int super_dim, typename T>
  static Vec<T, sub_dim> sub_vec( const Vec<T, super_dim> &vec,
                                  const Vec<uint8, super_dim> &selected )
  {
    Vec<T, sub_dim> sub_vec;
    int32 sub_axis = 0;
    for (int32 d = 0; d < super_dim && sub_axis < sub_dim; ++d)
      if (selected[d])
        sub_vec[sub_axis++] = vec[d];
    return sub_vec;
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
    return !panel_b.upwind_of_domain(source) &&
           panel_a.upwind_of_domain(source);
           
  }

  bool a_upwind_of_b(
      const StructuredDomainPanel &panel_a,
      const StructuredDomainPanel &panel_b,
      const Vec<Float, 3> &source)
  {
    return !panel_a.upwind_of_domain(source) &&
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

      const StructuredDomainPanel & operator[](int32 i);
      size_t size() const;

    protected:
      std::vector<StructuredDomainPanel> m_panels;
  };

  //--------------------------------------------------

  PanelList::PanelList()
  { }

  void PanelList::insert(const StructuredDomainPanel &panel)
  {
    m_panels.push_back(panel);
  }

  int32 PanelList::num_sockets(const UniformTopology *domain_mesh)
  {
    return StructuredDomainPanel::NUM_SIDES;
  }

  const StructuredDomainPanel & PanelList::operator[](int32 i)
  {
    return m_panels[i];
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
      while (range_end != planes.end() && 
             range_end->first == range_begin->first)
        ++range_end;
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

// separate all_panels by plane (orientation and sheet displacement)

// Digraph upwind;       // (i,j): j is upwind of i
// Digraph downwind;     // (i,j): j is downwind of i

// within a plane:
//   Generage list of AABB<2>
//   build tree
//   traverse for positive overlap (> cell_res/2):
//   for each pair of overlapping panels (A, B)
//           such that (upwind) domA -> A : B -> domB (downwind)
//     upwind.insert_edge(domB, domA);
//     downwind.insert_edge(domA, domB);




  }

  //==================================================================




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

  double sigmat_amplitude = 1;
  double sigmat_period = 1;
  if (config.m_config.has_child("sigmat_amplitude"))
    sigmat_amplitude = config.m_config["sigmat_amplitude"].to_double();
  if (config.m_config.has_child("sigmat_period"))
    sigmat_period = config.m_config["sigmat_period"].to_double();
  printf("sigmat_amplitude:%f\nsigmat_period:%f\n",
      sigmat_amplitude, sigmat_period);

  Vec<Float, 3> source = {{0.1, 0.1, 0.1}};

  Vec<Float, 3> global_origin = {{0, 0, 0}};
  Vec<Float, 3> spacing = {{1./64, 1./64, 1./64}};
  Vec<int32, 3> domains = {{4, 4, 4}};
  Vec<int32, 3> cell_dims = {{16, 16, 16}};

  const std::string field_name = "sigt";

  dray::Collection collection = egg_cartons(
      field_name,
      sigmat_amplitude,
      sigmat_period,
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


  // try to get domain connections
  {
    using namespace dray;
    typedef StructuredDomainPanel SDP;

    // Create panel_list.
    PanelList panel_list;
    const size_t num_domains = collection.local_size();
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
      domain_i++;
    }

    // Form graph from adjacent AABBs refering back to original indices
    Digraph domain_to_neighbor_panel;
    for (const std::pair<int32, int32> &ij : panel_list.adjacent_panels())
    {
      const int32 panel_i = ij.first,  panel_j = ij.second;
      const int32 domain_i = panel_list[panel_i].domain_idx();
      const int32 domain_j = panel_list[panel_j].domain_idx();

      // Make "undirected."
      domain_to_neighbor_panel.insert_edge(domain_i, panel_j);
      domain_to_neighbor_panel.insert_edge(domain_j, panel_i);

      // Assumes that each panel has at most one neighbor.
    }

    // for each source... (upwind/downwind depends on source)

    for (domain_i = 0; domain_i < num_domains; ++domain_i)
    {
      printf("\t[%d]", domain_to_neighbor_panel.count_edges(domain_i));
      if (domain_i == num_domains-1)
        printf("\n");

      //TODO
      // given domain, loop over connected sockets
      //   (neighbor_domain, socket) <- (domain, socket)
      //   // neighbor_panel <- panel
      //   test if (a_downwind_of_b(neighbor_panel, panel))
      //   test if (a_upwind_of_b(neighbor_panel, panel))
    }
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
      ///   UniformTopology * mesh = dynamic_cast<UniformTopology *>(domain.mesh());
      ///   LowOrderField * sigt = dynamic_cast<LowOrderField *>(domain.field(field_name));

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



