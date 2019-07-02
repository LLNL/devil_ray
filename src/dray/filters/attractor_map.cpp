#include <dray/filters/attractor_map.hpp>

#include <dray/GridFunction/mesh.hpp>
#include <dray/data_set.hpp>

#include <dray/array.hpp>
#include <dray/array_utils.hpp>
#include <dray/types.hpp>
#include <dray/vec.hpp>
#include <dray/math.hpp>

namespace dray
{

  class AttractorMapShader
  {
    protected:
      Vec4f u_inner_color;
      Vec4f u_outer_color;
      Vec4f u_edge_color;
      float32 u_inner_edge_radius_rcp;
      float32 u_outer_edge_radius_rcp;

    public:
      void set_uniforms(Vec4f inner_color,
                        Vec4f outer_color,
                        Vec4f edge_color,
                        float32 inner_edge_radius,
                        float32 outer_edge_radius)
      {
        u_inner_color = inner_color;
        u_outer_color = outer_color;
        u_edge_color = edge_color;

        u_inner_edge_radius_rcp = (inner_edge_radius > 0.0 ? 1.0 / inner_edge_radius : 0.05);
        u_outer_edge_radius_rcp = (outer_edge_radius > 0.0 ? 1.0 / outer_edge_radius : 3.0);
      }

      template <typename T>
      DRAY_EXEC static T convert_to_scalar(const Vec<T,3> &rcoords)
      {
        // 0 on element boundary, +0.5 in center, negative outside the element.
        T dist0 = 0.5 - fabs(rcoords[0] - 0.5);
        T dist1 = 0.5 - fabs(rcoords[1] - 0.5);
        T dist2 = 0.5 - fabs(rcoords[2] - 0.5);
        return min(min(dist0, dist1), dist2);
      }

      template <typename T>
      DRAY_EXEC Vec4f operator()(const Vec<T,3> &rcoords) const
      { 
        // For now, piecewise linear interpolation on the distance to nearest face.
        // TODO output a single channel for vtk image.
        T edge_dist = AttractorMapShader::convert_to_scalar<T>(rcoords);

        Vec4f color0 = u_edge_color;
        Vec4f color1 = (edge_dist >= 0.0 ? u_inner_color : u_outer_color);
        edge_dist = (edge_dist >= 0.0 ? edge_dist * u_inner_edge_radius_rcp :
                                        -edge_dist * u_outer_edge_radius_rcp);

        edge_dist = min(edge_dist, 1.0f);

        return color0 * (1 - edge_dist) + color1 * edge_dist;
      }
  };

  template<typename T>
  Array<Vec<float32,4>> AttractorMap::execute( const Vec<T,3> world_query_point,
                                               const Array<RefPoint<T,3>> &guesses,
                                               DataSet<T> &data_set)
  {
    using Color = Vec<float32, 4>;

    // Initialize color buffer.
    Array<Color> color_buffer;
    color_buffer.resize(guesses.size());

    // Initialize the color buffer to (0,0,0,0).
    const Color init_color = make_vec4f(0.f, 0.f, 0.f, 0.f);
    array_memset_vec(color_buffer, init_color);

    // Get mesh.
    const Mesh<T> &mesh = data_set.get_mesh();
    MeshAccess<T,3> device_mesh = mesh.access_device_mesh();

    // Set shader uniforms.
    AttractorMapShader shader;
    shader.set_uniforms({0,0,1,1}, {1,0,0,1}, {1,1,1,1}, 0.05, 3.0);

    Color *color_buffer_ptr = color_buffer.get_device_ptr();
    const RefPoint<T,3> *guess_ptr = guesses.get_device_ptr_const();

    RAJA::forall<for_policy>(RAJA::RangeSegment(0, guesses.size()), [=] DRAY_LAMBDA(const int32 sample_idx)
    {
      // Use ref_point.m_el_coords as initial guess, then receive into m_el_coords the solution.
      RefPoint<T,3> ref_point = guess_ptr[sample_idx];

      /// std::cout << "Before: " << ref_point.m_el_coords << "  ";

      device_mesh.world2ref(ref_point.m_el_id,
                            world_query_point,
                            ref_point.m_el_coords,
                            true);

      /// std::cout << "After: " << ref_point.m_el_coords << "\n";

      color_buffer_ptr[sample_idx] = shader(ref_point.m_el_coords);
    });

    return color_buffer;
  }



  template <typename T>
  Array<RefPoint<T,3>> AttractorMap::domain_grid_3d(uint32 grid_depth_x, uint32 grid_depth_y, uint32 grid_depth_z, int32 el_id)
  {
    const int32 grid_size_x = 1u << grid_depth_x;
    const int32 grid_size_y = 1u << grid_depth_y;
    const int32 grid_size_z = 1u << grid_depth_z;

    // The number of subintervals is the number of sample points - 1.
    const T grid_divisor_x = (grid_size_x - 1);
    const T grid_divisor_y = (grid_size_y - 1);
    const T grid_divisor_z = (grid_size_z - 1);

    const int32 total_num_samples = grid_size_x * grid_size_y * grid_size_z;

    Array<RefPoint<T,3>> guess_grid;
    guess_grid.resize(total_num_samples);
    RefPoint<T,3> *guess_grid_ptr = guess_grid.get_device_ptr();

    RAJA::forall<for_policy>(RAJA::RangeSegment(0, total_num_samples), [=] DRAY_LAMBDA (const int32 sample_idx)
    {
      // Index with x innermost and z outermost.
      const int32 xi = sample_idx & (grid_size_x - 1);
      const int32 yi = (sample_idx >> grid_depth_x) & (grid_size_y - 1);
      const int32 zi = (sample_idx >> grid_depth_x + grid_depth_y) /* & (grid_size_z - 1) */;

      guess_grid_ptr[sample_idx].m_el_id = el_id;
      guess_grid_ptr[sample_idx].m_el_coords = {((T) xi)/grid_divisor_x,
                                                ((T) yi)/grid_divisor_y,
                                                ((T) zi)/grid_divisor_z};
    });

    return guess_grid;
  }


  template <typename T>
  Array<RefPoint<T,3>> AttractorMap::domain_grid_slice_xy(uint32 grid_depth_x, uint32 grid_depth_y, T ref_z_val, int32 el_id)
  {
    const int32 grid_size_x = 1u << grid_depth_x;
    const int32 grid_size_y = 1u << grid_depth_y;

    // The number of subintervals is the number of sample points - 1.
    const T grid_divisor_x = (grid_size_x - 1);
    const T grid_divisor_y = (grid_size_y - 1);

    const int32 total_num_samples = grid_size_x * grid_size_y;

    Array<RefPoint<T,3>> guess_grid;
    guess_grid.resize(total_num_samples);
    RefPoint<T,3> *guess_grid_ptr = guess_grid.get_device_ptr();

    RAJA::forall<for_policy>(RAJA::RangeSegment(0, total_num_samples), [=] DRAY_LAMBDA (const int32 sample_idx)
    {
      // Index with x innermost and z outermost.
      const int32 xi = sample_idx & (grid_size_x - 1);
      const int32 yi = (sample_idx >> grid_depth_x) /* & (grid_size_y - 1) */;

      guess_grid_ptr[sample_idx].m_el_id = el_id;
      guess_grid_ptr[sample_idx].m_el_coords = {((T) xi)/grid_divisor_x,
                                                ((T) yi)/grid_divisor_y,
                                                ref_z_val};
    });

    return guess_grid;
  }



  //
  // Template instantiations.
  //

  template
  Array<Vec<float32,4>> AttractorMap::execute<float32>( const Vec<float32,3> world_query_point,
                                                        const Array<RefPoint<float32,3>> &guesses,
                                                        DataSet<float32> &data_set);

  template
  Array<Vec<float32,4>> AttractorMap::execute<float64>( const Vec<float64,3> world_query_point,
                                                        const Array<RefPoint<float64,3>> &guesses,
                                                        DataSet<float64> &data_set);

  template
  Array<RefPoint<float32,3>> AttractorMap::domain_grid_3d<float32>(uint32 grid_depth_x, uint32 grid_depth_y, uint32 grid_depth_z, int32 el_id);
  template
  Array<RefPoint<float64,3>> AttractorMap::domain_grid_3d<float64>(uint32 grid_depth_x, uint32 grid_depth_y, uint32 grid_depth_z, int32 el_id);

  template
  Array<RefPoint<float32,3>> AttractorMap::domain_grid_slice_xy<float32>(uint32 grid_depth_x, uint32 grid_depth_y, float32 ref_z_val, int32 el_id);
  template
  Array<RefPoint<float64,3>> AttractorMap::domain_grid_slice_xy<float64>(uint32 grid_depth_x, uint32 grid_depth_y, float64 ref_z_val, int32 el_id);


}//namespace dray
