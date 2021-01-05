#include <dray/filters/first_scatter.hpp>
#include <dray/uniform_topology.hpp>
#include <dray/error.hpp>
#include <dray/policies.hpp>
#include <dray/utils/point_writer.hpp>
#include <dray/utils/png_encoder.hpp>
#include <dray/GridFunction/low_order_field.hpp>
#include <dray/array_utils.hpp>
#include <dray/device_array.hpp>

namespace dray
{

namespace detail
{

// Returns cell center for ever cell in topo.
static Array<Vec<Float,3>> cell_centers(UniformTopology &topo)
{

  const Vec<int32,3> cell_dims = topo.cell_dims();
  const Vec<Float,3> origin = topo.origin();
  const Vec<Float,3> spacing = topo.spacing();

  const int32 num_cells = cell_dims[0] * cell_dims[1] * cell_dims[2];

  Array<Vec<Float,3>> locations;
  locations.resize(num_cells);
  Vec<Float,3> *loc_ptr = locations.get_device_ptr();

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, num_cells), [=] DRAY_LAMBDA (int32 index)
  {
    Vec<int32,3> cell_id;
    cell_id[0] = index % cell_dims[0];
    cell_id[1] = (index / cell_dims[0]) % cell_dims[1];
    cell_id[2] = index / (cell_dims[0] * cell_dims[1]);

    Vec<Float,3> loc;
    for(int32 i = 0; i < 3; ++i)
    {
      loc[i] = origin[i] + Float(cell_id[i]) * spacing[i] + spacing[i] * 0.5f;
    }

    loc_ptr[index] = loc;
  });

  return locations;
}

// Returns cell center of every cell in topo
// that has at least one nonzero emission value.
// The nonzero_list array (same length as return)
// contains the original indices of the nonzero cells.
static Array<Vec<Float,3>> cell_centers_nonzero(UniformTopology &topo,
                                                LowOrderField * emission,
                                                Array<int32> &nonzero_list)
{

  const Vec<int32,3> cell_dims = topo.cell_dims();
  const Vec<Float,3> origin = topo.origin();
  const Vec<Float,3> spacing = topo.spacing();

  nonzero_list = index_any_nonzero(emission->values());
  ConstDeviceArray<int32> nonzero_list_deva(nonzero_list);

  const int32 num_nonzero_cells = nonzero_list.size();

  Array<Vec<Float,3>> locations;
  locations.resize(num_nonzero_cells);
  Vec<Float,3> *loc_ptr = locations.get_device_ptr();

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, num_nonzero_cells),
      [=] DRAY_LAMBDA (int32 nz_index)
  {
    const int32 index = nonzero_list_deva.get_item(nz_index);
    Vec<int32,3> cell_id;
    cell_id[0] = index % cell_dims[0];
    cell_id[1] = (index / cell_dims[0]) % cell_dims[1];
    cell_id[2] = index / (cell_dims[0] * cell_dims[1]);

    Vec<Float,3> loc;
    for(int32 i = 0; i < 3; ++i)
    {
      loc[i] = origin[i] + Float(cell_id[i]) * spacing[i] + spacing[i] * 0.5f;
    }

    loc_ptr[nz_index] = loc;
  });

  return locations;
}

struct FS_TraversalState
{
  Vec<Float,3> m_delta_max;
  Vec<Float,3> m_delta;
  Vec<int32,3> m_voxel;
  Vec<Float,3> m_dir;

  // distance to voxel exit from initial point
  DRAY_EXEC
  Float exit() const
  {
    return min(m_delta_max[0], min(m_delta_max[1], m_delta_max[2]));
  }

  // advances to the next voxel along the ray
  DRAY_EXEC void advance()
  {
    int32 advance_dir = 0;
    for(int32 i = 1; i < 3; ++i)
    {
      if(m_delta_max[i] < m_delta_max[advance_dir])
      {
        advance_dir = i;
      }
    }
    m_delta_max[advance_dir] += m_delta[advance_dir];
    m_voxel[advance_dir] += m_dir[advance_dir] < 0.f ? -1 : 1;
  }

};

struct FS_DDATraversal
{
  const Vec<int32,3> m_dims;
  const Vec<Float,3> m_origin;
  const Vec<Float,3> m_spacing;

  FS_DDATraversal(UniformTopology &topo)
    : m_dims(topo.cell_dims()),
      m_origin(topo.origin()),
      m_spacing(topo.spacing())
  {

  }

  DRAY_EXEC
  bool is_inside(const Vec<int32, 3>& index) const
  {
    bool inside = true;
    const int32 minIndex = min(index[0], min(index[1], index[2]));
    if(minIndex < 0) inside = false;
    if(index[0] >= m_dims[0]) inside = false;
    if(index[1] >= m_dims[1]) inside = false;
    if(index[2] >= m_dims[2]) inside = false;
    return inside;
  }

  DRAY_EXEC
  int32 voxel_index(const Vec<int32, 3> &voxel) const
  {
    return voxel[0] + voxel[1] * m_dims[0] + voxel[2] * m_dims[0] * m_dims[1];
  }

  DRAY_EXEC Float
  init_traversal(const Vec<Float,3> &point,
                 const Vec<Float,3> &dir,
                 FS_TraversalState &state) const
  {
    Vec<Float, 3> temp = point;
    temp = temp - m_origin;
    state.m_voxel[0] = temp[0] / m_spacing[0];
    state.m_voxel[1] = temp[1] / m_spacing[1];
    state.m_voxel[2] = temp[2] / m_spacing[2];
    state.m_dir = dir;

    Vec<Float,3> step;
    step[0] = (dir[0] >= 0.f) ? 1.f : -1.f;
    step[1] = (dir[1] >= 0.f) ? 1.f : -1.f;
    step[2] = (dir[2] >= 0.f) ? 1.f : -1.f;

    Vec<Float,3> next_boundary;
    next_boundary[0] = (Float(state.m_voxel[0]) + step[0]) * m_spacing[0];
    next_boundary[1] = (Float(state.m_voxel[1]) + step[1]) * m_spacing[1];
    next_boundary[2] = (Float(state.m_voxel[2]) + step[2]) * m_spacing[2];

    // correct next boundary for negative directions
    if(step[0] == -1.f) next_boundary[0] += m_spacing[0];
    if(step[1] == -1.f) next_boundary[1] += m_spacing[1];
    if(step[2] == -1.f) next_boundary[2] += m_spacing[2];

    // distance to next voxel boundary
    state.m_delta_max[0] = (dir[0] != 0.f) ?
      (next_boundary[0] - (point[0] - m_origin[0])) / dir[0] : infinity<Float>();

    state.m_delta_max[1] = (dir[1] != 0.f) ?
      (next_boundary[1] - (point[1] - m_origin[1])) / dir[1] : infinity<Float>();

    state.m_delta_max[2] = (dir[2] != 0.f) ?
      (next_boundary[2] - (point[2] - m_origin[2])) / dir[2] : infinity<Float>();

    // distance along ray to traverse x,y, and z of a voxel
    state.m_delta[0] = (dir[0] != 0) ? m_spacing[0] / dir[0] * step[0] : infinity<Float>();
    state.m_delta[1] = (dir[1] != 0) ? m_spacing[1] / dir[1] * step[1] : infinity<Float>();
    state.m_delta[2] = (dir[2] != 0) ? m_spacing[2] / dir[2] * step[2] : infinity<Float>();

    Vec<Float,3> exit_boundary;
    exit_boundary[0] = step[0] < 0.f ? 0.f : Float(m_dims[0]) * m_spacing[0];
    exit_boundary[1] = step[1] < 0.f ? 0.f : Float(m_dims[1]) * m_spacing[1];
    exit_boundary[2] = step[2] < 0.f ? 0.f : Float(m_dims[2]) * m_spacing[2];

    // Masado questions these lines
    if(step[0] == -1.f) exit_boundary[0] += m_spacing[0];
    if(step[1] == -1.f) exit_boundary[1] += m_spacing[1];
    if(step[2] == -1.f) exit_boundary[2] += m_spacing[2];

    Vec<Float,3> exit_dist;
    // distance to grid exit
    exit_dist[0] = (dir[0] != 0.f) ?
      (exit_boundary[0] - (point[0] - m_origin[0])) / dir[0] : infinity<Float>();

    exit_dist[1] = (dir[1] != 0.f) ?
      (exit_boundary[1] - (point[1] - m_origin[1])) / dir[1] : infinity<Float>();

    exit_dist[2] = (dir[2] != 0.f) ?
      (exit_boundary[2] - (point[2] - m_origin[2])) / dir[2] : infinity<Float>();

    //std::cout<<"Init voxel "<<voxel<<"\n";

    return min(exit_dist[0], min(exit_dist[1], exit_dist[2]));
  }



};


} // namespace detail

FirstScatter::FirstScatter()
{

}


// Returns flattened array of size num_destinations * num_sources.
// Results for each source and destination, with sources varying faster.
static
Array<Float>
go_trace(Array<Vec<Float,3>> &destinations,
         Array<Vec<Float,3>> &ray_sources,
         UniformTopology &topo,
         LowOrderField *absorption)
{
  // input
  const detail::FS_DDATraversal dda(topo);
  const Vec<Float,3> *destn_ptr = destinations.get_device_ptr_const();
  const Vec<Float,3> *ray_src_ptr = ray_sources.get_device_ptr_const();
  const int32 size_ray_srcs = ray_sources.size();
  const int32 size = destinations.size();
  const ConstDeviceArray<Float> absorption_arr( absorption->values() );

  const int32 ncomp = absorption_arr.ncomp();

  // output
  Array<Float> path_lengths;
  path_lengths.resize(size * size_ray_srcs, ncomp);
  NonConstDeviceArray<Float> length_arr( path_lengths );

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, size), [=] DRAY_LAMBDA (int32 index)
  {
    Float * res = new Float[ncomp];

    Vec<Float,3> destn = destn_ptr[index];
    for(int ray_src = 0; ray_src < size_ray_srcs; ++ray_src)
    {
      Vec<Float,3> loc = ray_src_ptr[ray_src];
      Vec<Float,3> dir = destn - loc;
      dir.normalize();
      detail::FS_TraversalState state;
      dda.init_traversal(loc, dir, state);

      Float distance = 0.f;
      for (int32 component = 0; component < ncomp; ++component)
        res[component] = 1.f;

      while(dda.is_inside(state.m_voxel))
      {
        const Float voxel_exit = state.exit();
        const Float length = voxel_exit - distance;

        const int32 cell_id = dda.voxel_index(state.m_voxel);
        for (int32 component = 0; component < ncomp; ++component)
        {
          const Float absorb
              = exp(-absorption_arr.get_item(cell_id, component) * length);
          res[component] = res[component] * absorb;
        }
        // this will get more complicated with MPI and messed up
        // metis domain decompositions

        distance = voxel_exit;
        state.advance();
      }
      // Directions matter.
      // Instead of summing over all sources,
      // return result from each source separately.
      for (int32 component = 0; component < ncomp; ++component)
        length_arr.get_item(size_ray_srcs * index + ray_src, component)
            = res[component];
    }

    delete [] res;
  });
  return path_lengths;
}


Array<Float> integrate_moments(Array<Vec<Float,3>> &destinations,
                               int32 legendre_order,
                               Array<Float> &path_lengths,
                               Array<Vec<Float,3>> &ray_sources,
                               Array<int32> &source_cells,
                               Float _cell_volume,
                               LowOrderField *emission);



void FirstScatter::execute(DataSet &data_set)
{
  if(m_total_cross_section_field == "")
  {
    DRAY_ERROR("Total cross section field not set");
  }

  if(m_emission_field == "")
  {
    DRAY_ERROR("Emission field not set");
  }

  if(!data_set.has_field(m_total_cross_section_field))
  {
    DRAY_ERROR("No total cross section field '"<<m_total_cross_section_field<<"' found");
  }

  if(!data_set.has_field(m_emission_field))
  {
    DRAY_ERROR("No emission field '"<<m_emission_field<<"' found");
  }

  TopologyBase *topo = data_set.topology();
  if(dynamic_cast<UniformTopology*>(topo) != nullptr)
  {
    std::cout<<"Boom\n";

    UniformTopology *uni_topo = dynamic_cast<UniformTopology*>(topo);
    LowOrderField *total_cross_section = dynamic_cast<LowOrderField*>(data_set.field(m_total_cross_section_field));
    LowOrderField *emission = dynamic_cast<LowOrderField*>(data_set.field(m_emission_field));

    if(total_cross_section->assoc() != LowOrderField::Assoc::Element)
    {
      DRAY_ERROR("Total cross section field must be associated with elements");
    }

    if(emission->assoc() != LowOrderField::Assoc::Element)
    {
      DRAY_ERROR("Emission field must be associated with elements");
    }

    const int32 legendre_order = 3;  // TODO get legendre_order from dataset?

    Array<int32> source_cells;
    Array<Vec<Float,3>> ray_sources = detail::cell_centers_nonzero(*uni_topo, emission, source_cells);
    Array<Vec<Float,3>> destinations = detail::cell_centers(*uni_topo);
    Array<Float> plengths = go_trace(destinations, ray_sources, *uni_topo, total_cross_section);

    const Float cell_volume = (uni_topo->spacing()[0]
                               * uni_topo->spacing()[1]
                               * uni_topo->spacing()[2]);

    Array<Float> destination_moments = integrate_moments(destinations,
                                                         legendre_order,
                                                         plengths,
                                                         ray_sources,
                                                         source_cells,
                                                         cell_volume,
                                                         emission);

    //TODO return destination_moments

    std::cout << "destinations.size() == " << destinations.size() << "\n";
    std::cout << "legendre_order == " << legendre_order << "\n";
    std::cout << "num_moments == " << (legendre_order+1)*(legendre_order+1) << "\n";
    std::cout << "destination_moments.size() == " << destination_moments.size() << "\n";
  }
  else
  {
    DRAY_ERROR("FirstScatter filter only supports UniformTopology");
  }
}

void FirstScatter::total_cross_section_field(const std::string field_name)
{
  m_total_cross_section_field = field_name;
}

void FirstScatter::emission_field(const std::string field_name)
{
  m_emission_field = field_name;
}



class SphericalHarmonics
{
  public:
    DRAY_EXEC SphericalHarmonics(int legendre_order) : m_legendre_order(legendre_order) {}
    DRAY_EXEC ~SphericalHarmonics()
    {
      delete_buffer();
    }

    /** Evaluates all spherical harmonics up to legendre_order. */
    template <typename T>
    DRAY_EXEC
    const T* eval_all(const dray::Vec<T, 3> &xyz_normal);

    /** Calls eval_all() and performs dot product. */
    template <typename T>
    DRAY_EXEC
    T eval_function(const T * coefficients, const dray::Vec<T, 3> &xyz_normal)
    {
      return eval_function<T>(m_legendre_order,
                              coefficients,
                              eval_all(xyz_normal));
    }

    /** Calls eval_all() and accumulates vector to coefficients. */
    template <typename T>
    DRAY_EXEC
    void project_point(T * coefficients,
                       const dray::Vec<T, 3> &xyz_normal,
                       const T integration_value,
                       const T integration_weight)
    {
      project_point<T>(m_legendre_order,
                       coefficients,
                       eval_all(xyz_normal),
                       integration_value,
                       integration_weight);
    }

    DRAY_EXEC int num_harmonics() const { return num_harmonics(m_legendre_order); }


    DRAY_EXEC static int index(int n, int m) { return n * (n+1) + m; }
    DRAY_EXEC static int alp_index(int n, int m) { return n * (n+1) / 2 + m; }
    // alp = associated legendre polynomial, only uses m >= 0.

    DRAY_EXEC static int num_harmonics(int legendre_order)
    {
      return (legendre_order+1)*(legendre_order+1);
    }

    /** Static version does not call eval_all().
     *  Good for evaluating different functions
     *  with different sets of coefficients. */
    template <typename T>
    DRAY_EXEC
    static T eval_function(const int legendre_order,
                         const T * coefficients,
                         const T * sph_harmonics)
    {
      T value = 0.0f;
      const int Np1_sq = num_harmonics(legendre_order);
      for (int nm = 0; nm < Np1_sq; ++nm)
        value += coefficients[nm] * sph_harmonics[nm];
      return value;
    }

    /** Static version does not call eval_all().
     *  Good for projecting different integration values
     *  to different sets of coefficients. */
    template <typename T>
    DRAY_EXEC
    static void project_point(const int legendre_order,
                              T * coefficients,
                              const T * sph_harmonics,
                              const T integration_value,
                              const T integration_weight)
    {
      const int Np1_sq = num_harmonics(legendre_order);
      const T integration_product = integration_value * integration_weight;
      for (int nm = 0; nm < Np1_sq; ++nm)
        coefficients[nm] += sph_harmonics[nm] * integration_product;
    }

  private:
    template <typename T>
    DRAY_EXEC T * resize_buffer(const size_t size);
    DRAY_EXEC void delete_buffer();

  private:
    int m_legendre_order = 0;
    size_t m_buffer_size = 0;
    char * m_buffer = nullptr;
};


Array<Float> integrate_moments(Array<Vec<Float,3>> &destinations,
                               int32 _legendre_order,
                               Array<Float> &path_lengths,
                               Array<Vec<Float,3>> &ray_sources,
                               Array<int32> &source_cells,
                               Float _cell_volume,
                               LowOrderField *emission)
{
  using sph_t = Float;

  const int32 ncomp = path_lengths.ncomp();
  const int32 legendre_order = _legendre_order;
  const int32 num_moments = (legendre_order + 1) * (legendre_order + 1);
  const int32 num_destinations = destinations.size();
  const int32 num_sources = ray_sources.size();

  Array<Float> destination_moments;
  destination_moments.resize(num_destinations * num_moments, ncomp);

  const Float cell_volume = _cell_volume;

  ConstDeviceArray<Vec<Float, 3>> destinations_dev(destinations);
  ConstDeviceArray<Vec<Float, 3>> ray_sources_dev(ray_sources);
  ConstDeviceArray<Float> path_lengths_dev(path_lengths);
  ConstDeviceArray<int32> source_cells_dev(source_cells);
  ConstDeviceArray<Float> emission_dev(emission->values());

  NonConstDeviceArray<Float> destination_moments_dev(destination_moments);

  RAJA::forall<for_policy> (RAJA::RangeSegment(0, num_destinations),
      [=] DRAY_LAMBDA (int32 dest)
  {
    // Clear output.
    for (int32 nm = 0; nm < num_moments; ++nm)
      for (int32 component = 0; component < ncomp; ++component)
        destination_moments_dev.get_item(num_moments * dest + nm, component) = 0.0f;

    SphericalHarmonics sph(legendre_order);

    // For each source
    //   For each component
    //     For each moment
    //       Multiply-and-accumulate source term with spherical harmonic.
    const Vec<Float, 3> dest_pos = destinations_dev.get_item(dest);
    for (int32 source = 0; source < num_sources; ++source)
    {
      const Vec<Float, 3> omega = (dest_pos - ray_sources_dev.get_item(source));
      const Vec<Float, 3> omega_hat = omega.normalized();
      const Float rcp_mag2 = rcp_safe(omega.magnitude2());
      // Really should use volume-average (over source cell) of rcp_mag2.

      const sph_t * sph_eval = sph.eval_all<sph_t>(omega_hat);

      const int32 source_idx = source_cells_dev.get_item(source);

      for (int32 component = 0; component < ncomp; ++component)
      {
        const Float dEmission_dV = emission_dev.get_item(source_idx, component);
        const Float source_dL_dOmega
          = dEmission_dV * cell_volume * rcp_mag2;

        const Float transmitted = path_lengths_dev.get_item(
            num_sources * dest + source, component);

        const Float trans_source = transmitted * source_dL_dOmega;

        for (int32 nm = 0; nm < num_moments; ++nm)
        {
          const sph_t spherical_harmonic = sph_eval[nm];
          destination_moments_dev.get_item(num_moments * dest + nm, component)
              += spherical_harmonic * trans_source;
        }//moments
      }//components
    }//sources
  });//destinations

  return destination_moments;
}


template <typename T>
DRAY_EXEC
T * SphericalHarmonics::resize_buffer(const size_t size)
{
  size_t new_size = sizeof(T) * size;
  if (m_buffer_size < new_size)
  {
    if (m_buffer != nullptr)
      delete [] m_buffer;
    m_buffer = new char[new_size];
  }
  return (T*)(m_buffer);
}


DRAY_EXEC
void SphericalHarmonics::delete_buffer()
{
  if (m_buffer != nullptr)
    delete [] m_buffer;
}

template <typename T>
DRAY_EXEC
const T* SphericalHarmonics::eval_all(const dray::Vec<T, 3> &xyz_normal)
{
  // Computed using the recursive formulation in Appendix A1 in
  //
  //     @inproceedings{sloan2008stupid,
  //       title={Stupid spherical harmonics (sh) tricks},
  //       author={Sloan, Peter-Pike},
  //       booktitle={Game developers conference},
  //       volume={9},
  //       pages={42},
  //       year={2008}
  //     }

  // Note: I came up with a recursive form of the normalization constants K_n^m.
  //   The formula for K_n^m involves ratios of factorials. I used floats
  //   because the ratios do not simply to integers. I haven't studied the stability
  //   properties of evaluating them directly or recursively, so no guarantees.
  //   Also, to test the normalization constants you need to do a reconstruction,
  //   not just evaluate each spherical harmonic individually.

  const int Np1 = m_legendre_order + 1;
  const int Np1_sq = Np1 * Np1;
  const int result_sz = Np1_sq;            // result
  const int sin_sz = Np1;                  // sine
  const int cos_sz = Np1;                  // cosine
  const int alp_sz = Np1 * (Np1+1) / 2;    // associated legendre polynomial
  const int k2_sz = Np1 * (Np1+1) / 2;     // square of normalization constant

  T * const buffer = resize_buffer<T>(result_sz + sin_sz + cos_sz + alp_sz + k2_sz);

  T * const resultp = buffer;
  T * const sinp = resultp + result_sz;
  T * const cosp = sinp + sin_sz;
  T * const alpp = cosp + cos_sz;
  T * const k2p = alpp + alp_sz;

  const T sqrt2 = sqrtl(2);

  const T &x = xyz_normal[0];
  const T &y = xyz_normal[1];
  const T &z = xyz_normal[2];

  // m=0
  {
    const int m = 0;

    sinp[m] = 0;
    cosp[m] = 1;

    // n == m
    alpp[alp_index(m, m)] = 1;
    k2p[alp_index(0, 0)] = 1.0 / (4 * dray::pi());
    resultp[index(m, m)] = sqrt(k2p[alp_index(m, m)]) * alpp[alp_index(m, m)];
    /// resultp[index(m, m)] = Knm(m, m) * alpp[alp_index(m, m)];

    // n == m+1
    if (m+1 <= m_legendre_order)
    {
      alpp[alp_index(m+1, m)] = (2*m+1) * z * alpp[alp_index(m, m)];
      k2p[alp_index(1, 0)] = 2 * (1+1) / (4 * dray::pi());
      resultp[index(m+1, m)] = sqrt(k2p[alp_index(m+1, m)]) * alpp[alp_index(m+1, m)];
      /// resultp[index(m+1, m)] = Knm(m+1, m) * alpp[alp_index(m+1, m)];
    }

    // n >= m+2
    for (int n = m+2; n <= m_legendre_order; ++n)
    {
      alpp[alp_index(n, m)] = ( (2*n-1) * z * alpp[alp_index(n-1, m)]
                               -(n+m-1)     * alpp[alp_index(n-2, m)] ) / (n-m);

      k2p[alp_index(n, 0)] = (2*n+1) / (4 * dray::pi());

      resultp[index(n, m)] = sqrt(k2p[alp_index(n, m)]) * alpp[alp_index(n, m)];
      /// resultp[index(n, m)] = Knm(n, m) * alpp[alp_index(n, m)];
    }
  }

  // m>0
  for (int m = 1; m <= m_legendre_order; ++m)
  {
    sinp[m] = x * sinp[m-1] + y * cosp[m-1];
    cosp[m] = x * cosp[m-1] - y * sinp[m-1];

    // n == m
    alpp[alp_index(m, m)] = (1-2*m) * alpp[alp_index(m-1, m-1)];;
    k2p[alp_index(m, m)] = k2p[alp_index(m-1, m-1)] * (2*m+1) / ((2*m-1) * (2*m-1) * (2*m));
    resultp[index(m, m)] = sqrt(2*k2p[alp_index(m, m)]) * cosp[m] * alpp[alp_index(m, m)];
    /// resultp[index(m, m)] = sqrt2*Knm(m, m) * cosp[m] * alpp[alp_index(m, m)];

    // n == m+1
    if (m+1 <= m_legendre_order)
    {
      alpp[alp_index(m+1, m)] = (2*m+1) * z * alpp[alp_index(m, m)];
      k2p[alp_index(m+1, m)] =
          k2p[alp_index((m+1)-1, m)] * (2*(m+1)+1) * ((m+1)-m) / ((2*(m+1)-1) * ((m+1)+m));

      resultp[index(m+1, m)] = sqrt(2*k2p[alp_index(m+1, m)]) * cosp[m] * alpp[alp_index(m+1, m)];
      /// resultp[index(m+1, m)] = sqrt2*Knm(m+1, m) * cosp[m] * alpp[alp_index(m+1, m)];
    }

    // n >= m+2
    for (int n = m+2; n <= m_legendre_order; ++n)
    {
      alpp[alp_index(n, m)] = ( (2*n-1) * z * alpp[alp_index(n-1, m)]
                               -(n+m-1)     * alpp[alp_index(n-2, m)] ) / (n-m);

      k2p[alp_index(n, m)] = k2p[alp_index(n-1, m)] * 2*(n+1) * (n-m) / ((2*n-1) * (n+m));

      resultp[index(n, m)] = sqrt(2*k2p[alp_index(n, m)]) * cosp[m] * alpp[alp_index(n, m)];
      /// resultp[index(n, m)] = sqrt2*Knm(n, m) * cosp[m] * alpp[alp_index(n, m)];
    }
  }

  // m<0
  for (int m = -1; m >= -m_legendre_order; --m)
  {
    const int absm = -m;
    for (int n = absm; n <= m_legendre_order; ++n)
    {
      resultp[index(n, m)] = sqrt(2*k2p[alp_index(n, absm)]) * sinp[absm] * alpp[alp_index(n, absm)];
      /// resultp[index(n, m)] = sqrt2*Knm(n, absm) * sinp[absm] * alpp[alp_index(n, absm)];
    }
  }

  return resultp;
}




};//namespace dray

