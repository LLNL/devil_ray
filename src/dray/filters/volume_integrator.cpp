#include <dray/filters/volume_integrator.hpp>
#include <dray/filters/internal/get_fragments.hpp>
#include <dray/GridFunction/device_mesh.hpp>
#include <dray/GridFunction/device_field.hpp>
#include <dray/device_framebuffer.hpp>
#include <dray/device_color_map.hpp>
#include <dray/array_utils.hpp>
#include <dray/shaders.hpp>
#include <dray/utils/data_logger.hpp>

#include <assert.h>

//move to dispatcher
#include<dray/derived_topology.hpp>
#include<dray/derived_field.hpp>

namespace dray
{

namespace detail
{

template<typename MeshType, typename FieldType>
DRAY_EXEC
void scalar_gradient(const Location &loc,
                     MeshType &mesh,
                     FieldType &field,
                     Float &scalar,
                     Vec<Float,3> &gradient)
{

  // i think we need this to oreient the deriv
  Vec<Vec<Float, 3>, 3> jac_vec;
  Vec<Float, 3> world_pos = // don't need this but we need the jac
    mesh.get_elem(loc.m_cell_id).eval_d(loc.m_ref_pt, jac_vec);

  Vec<Vec<Float, 1>, 3> field_deriv;
  scalar =
    field.get_elem(loc.m_cell_id).eval_d(loc.m_ref_pt, field_deriv)[0];

  Matrix<Float, 3, 3> jacobian_matrix;
  Matrix<Float, 1, 3> gradient_ref;
  for(int32 rdim = 0; rdim < 3; ++rdim)
  {
    jacobian_matrix.set_col(rdim, jac_vec[rdim]);
    gradient_ref.set_col(rdim, field_deriv[rdim]);
  }

  bool inv_valid;
  const Matrix<Float, 3, 3> j_inv = matrix_inverse(jacobian_matrix, inv_valid);
  //TODO How to handle the case that inv_valid == false?
  const Matrix<Float, 1, 3> gradient_mat = gradient_ref * j_inv;
  gradient = gradient_mat.get_row(0);
}

} // namespace detail

VolumeIntegrator::VolumeIntegrator()
  : m_color_table("ColdAndHot"),
    m_num_samples(100)
{
  m_color_table.add_alpha(0.0000, .1f);
  m_color_table.add_alpha(1.0000, .2f);
}

template<typename Topology, typename Functor>
void dispatch_scalar_field(FieldBase *field, Topology *topo, Functor &func)
{
  if(dynamic_cast<Field<Hex1>*>(field) != nullptr)
  {
    Field<Hex1>* hex1 = dynamic_cast<Field<Hex1>*>(field);
    func(*topo, *hex1);
    std::cout<<"hex field 1\n";
  }
  else if(dynamic_cast<Quad1*>(field) != nullptr)
  {
    std::cout<<"quad field 1\n";
  }
  else
  {
    std::cout<<"field cast failed\n";
  }
}

template<typename Functor>
void dispatch_3d(TopologyBase *topo, FieldBase *field, Functor &func)
{
  if(dynamic_cast<HexTopology*>(topo) != nullptr)
  {
    HexTopology *hex_topo = dynamic_cast<HexTopology*>(topo);
    std::cout<<"hex 1\n";
    dispatch_scalar_field(field, hex_topo, func);
  }
  else
  {
    // we don't support this type
  }
}

struct Functor
{
  VolumeIntegrator *m_integrator;
  Array<Ray> *m_rays;
  Framebuffer *m_fb;
  Functor(VolumeIntegrator *integrator,
          Array<Ray> *rays,
          Framebuffer *fb)
    : m_integrator(integrator),
      m_rays(rays),
      m_fb(fb)
  {
  }

  template<typename TopologyType, typename FieldType>
  void operator()(TopologyType &topo, FieldType &field)
  {
    m_integrator->trace(topo.mesh(), field, *m_rays, *m_fb);
  }
};

template<typename MeshElem, typename FieldElem>
void VolumeIntegrator::trace(Mesh<MeshElem> &mesh,
                             Field<FieldElem> &field,
                             Array<Ray> &rays,
                             Framebuffer &fb)
{
  DRAY_LOG_OPEN("volume");

  assert(m_field_name != "");

  constexpr float32 correction_scalar = 10.f;
  float32 ratio = correction_scalar / m_num_samples;
  ColorTable corrected = m_color_table.correct_opacity(ratio);

  //Field<FieldOn<ElemT, 1u>> field = data_set.get_field(m_field_name);

  dray::AABB<> bounds = mesh.get_bounds();
  dray::float32 mag = (bounds.max() - bounds.min()).magnitude();
  const float32 sample_dist = mag / dray::float32(m_num_samples);

  const int32 num_elems = mesh.get_num_elem();

  DRAY_LOG_ENTRY("samples", m_num_samples);
  DRAY_LOG_ENTRY("sample_distance", sample_dist);
  DRAY_LOG_ENTRY("cells", num_elems);
  // Start the rays out at the min distance from calc ray start.
  // Note: Rays that have missed the mesh bounds will have near >= far,
  //       so after the copy, we can detect misses as dist >= far.

  // Initial compaction: Literally remove the rays which totally miss the mesh.
  cull_missed_rays(rays, mesh.get_bounds());

  Array<RefPoint<3>> rpoints;
  rpoints.resize(rays.size());

  const RefPoint<3> invalid_refpt{ -1, {-1,-1,-1} };

  const int32 ray_size = rays.size();
  const Ray *rays_ptr = rays.get_device_ptr_const();

  // complicated device stuff
  DeviceMesh<MeshElem> device_mesh(mesh);
  DeviceFramebuffer d_framebuffer(fb);
  //FieldAccess<FieldOn<ElemT, 1u>> device_field = field.access_device_field();
  DeviceField<FieldElem> device_field(field);

  //Colors!
  ColorMap color_map;
  color_map.color_table(corrected);
  color_map.scalar_range(field.get_range());
  DeviceColorMap d_color_map(color_map);

  //Array<stats::Stats> mstats;
  //mstats.resize(size);
  //stats::Stats *mstats_ptr = mstats.get_device_ptr();

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, ray_size), [=] DRAY_LAMBDA (int32 i)
  {
    const Ray ray = rays_ptr[i];
    // advance the ray one step
    Float distance = ray.m_near + sample_dist;
    Vec4f color = d_framebuffer.m_colors[ray.m_pixel_id];

    while(distance < ray.m_far)
    {
      //Vec<Float,3> point = ray.m_orig + ray.m_dir * distance;
      Vec<Float,3> point = ray.m_orig + distance * ray.m_dir;
      Location loc = device_mesh.locate(point);

      if(loc.m_cell_id != -1)
      {
        Vec<Float,3> gradient;
        Float scalar;
        detail::scalar_gradient(loc, device_mesh, device_field, scalar, gradient);
        Vec4f sample_color = d_color_map.color(scalar);

        //composite
        sample_color[3] *= (1.f - color[3]);
        color[0] = color[0] + sample_color[0] * sample_color[3];
        color[1] = color[1] + sample_color[1] * sample_color[3];
        color[2] = color[2] + sample_color[2] * sample_color[3];
        color[3] = sample_color[3] + color[3];
        if(color[3] > 0.95f)
        {
          // terminate
          distance = ray.m_far;
        }
      }

      distance += sample_dist;
    }
    d_framebuffer.m_colors[ray.m_pixel_id] = color;
    // should this be first valid sample or even set this?
    //d_framebuffer.m_depths[pid] = hit.m_dist;

  });

  fb.composite_background();
  DRAY_LOG_CLOSE();
}

void
VolumeIntegrator::execute(Array<Ray> &rays,
                          nDataSet &data_set,
                          Framebuffer &fb)
{

  assert(m_field_name != "");

  TopologyBase *topo = data_set.topology();
  FieldBase *field = data_set.field(m_field_name);

  Functor func(this, &rays, &fb);
  dispatch_3d(topo, field, func);
#if 0
#endif
}

void
VolumeIntegrator::set_field(const std::string field_name)
{
 m_field_name = field_name;
}

void
VolumeIntegrator::set_color_table(const ColorTable &color_table)
{
  m_color_table = color_table;
}

void
VolumeIntegrator::set_num_samples(const int32 num_samples)
{
  assert(num_samples > 0);
  m_num_samples = num_samples;
}

//using Hex = MeshElem<3u, ElemType::Quad, Order::General>;
//template
//void
//VolumeIntegrator::execute<Hex>(Array<Ray> &rays,
//                               DataSet<Hex> &data_set,
//                               Framebuffer &fb);

}//namespace dray

