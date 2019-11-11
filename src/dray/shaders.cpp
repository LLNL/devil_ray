#include <dray/shaders.hpp>

#include <dray/math.hpp>
#include <dray/policies.hpp>
#include <dray/device_framebuffer.hpp>
#include <dray/device_color_map.hpp>

namespace dray
{
// init static members
int32 Shader::m_color_samples = 1024;
//Array<Vec4f> Shader::m_color_map;
ColorTable Shader::m_color_table;
PointLightSource Shader::m_light = {{20.f, 10.f, 50.f},
                                    {0.1f, 0.1f, 0.1f},
                                    {0.3f, 0.3f, 0.3f},
                                    {0.7f, 0.7f, 0.7f},
                                    80.0 };


void
Shader::composite_bg(dray::Array<dray::Vec<float, 4> > &color_buffer,
                     const dray::Vec<float, 4> &bg_color)
{
  // avoid lambda capture issues
  Vec4f background = bg_color;
  Vec4f *img_ptr = color_buffer.get_device_ptr();
  const int32 size = color_buffer.size();

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, size), [=] DRAY_LAMBDA (int32 i)
  {
    Vec4f color = img_ptr[i];
    if(color[3] < 1.f)
    {
      //composite
      float32 alpha = background[3] * (1.f - color[3]);
      color[0] = color[0] + background[0] * alpha;
      color[1] = color[1] + background[1] * alpha;
      color[2] = color[2] + background[2] * alpha;
      color[3] = alpha + color[3];
      img_ptr[i] = color;
    }
  });
} // composite bg

void
Shader::set_color_table(const ColorTable &color_table)
{
  //color_table.sample(m_color_samples, m_color_map);
  m_color_table = color_table;
} // set_color table

void
Shader::blend(Framebuffer &fb,
              ColorMap &color_map,
              const Array<Ray> &rays,
              const Array<RayHit> &hits,
              const Array<Fragment> &fragments)

{
  assert(rays.size() == fragments.size());
  assert(hits.size() == fragments.size());


  DeviceColorMap d_color_map(color_map);
  DeviceFramebuffer d_framebuffer(fb);

  const Ray *ray_ptr = rays.get_device_ptr_const();
  const RayHit *hit_ptr = hits.get_device_ptr_const();
  const Fragment *frag_ptr = fragments.get_device_ptr_const();

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, rays.size()), [=] DRAY_LAMBDA (int32 ii)
  {
    const Ray ray = ray_ptr[ii];
    const RayHit hit = hit_ptr[ii];
    const Fragment fragment = frag_ptr[ii];

    if(hit.m_hit_idx > -1)
    {
      const int32 pid = ray.m_pixel_id;
      const Float sample_val = fragment.m_scalar;

      Vec4f sample_color = d_color_map.color(sample_val);

      Vec4f color = d_framebuffer.m_colors[pid];
      //composite
      sample_color[3] *= (1.f - color[3]);
      color[0] = color[0] + sample_color[0] * sample_color[3];
      color[1] = color[1] + sample_color[1] * sample_color[3];
      color[2] = color[2] + sample_color[2] * sample_color[3];
      color[3] = sample_color[3] + color[3];
      d_framebuffer.m_colors[pid] = color;
      d_framebuffer.m_depths[pid] = hit.m_dist;
      //std::cout<<"sample color "<<sample_color<<" "<<sample_val<<" "<<color<<"\n";
    }
  });
}//blend


void Shader::blend_surf(Framebuffer &fb,
                        ColorMap &color_map,
                        const Array<Ray> &rays,
                        const Array<RayHit> &hits,
                        const Array<Fragment> &fragments)

{
  // TODO: i feel like we either go with a shading context
  // that is just populated with RayHits/Locations + rays
  // Slice is a example of something that we get distances to the
  // plane in a different way, but we are currenlty missibg world
  // points. One way is to get ray hits that are specially populated,
  // another way is to just copy everything to a shading context.
  // So far I think we should just populate ray hit in slice since
  // that is what we are technically looking for anyway (also
  // solved the distance issue
  const RayHit * hit_ptr  = hits.get_device_ptr_const();
  const Ray * ray_ptr  = rays.get_device_ptr_const();
  const Fragment * frag_ptr  = fragments.get_device_ptr_const();

  DeviceFramebuffer d_framebuffer(fb);
  DeviceColorMap d_color_map(color_map);

  const Vec<Float,3> light_pos = {m_light.m_pos[0], m_light.m_pos[1], m_light.m_pos[2]};
    // Local for lambda.
  const Vec<float32,3> &light_amb = m_light.m_amb;
  const Vec<float32,3> &light_diff = m_light.m_diff;
  const Vec<float32,3> &light_spec = m_light.m_spec;
  const float32 &spec_pow = m_light.m_spec_pow; //shiny

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, hits.size()), [=] DRAY_LAMBDA (int32 ii)
  {
    const RayHit &hit = hit_ptr[ii];
    const Fragment &frag = frag_ptr[ii];
    const Ray &ray = ray_ptr[ii];
    if (hit.m_hit_idx > -1)
    {
      const int32 pid = ray.m_pixel_id;
      const Float sample_val = frag.m_scalar;

      Vec4f sample_color = d_color_map.color(sample_val);

      Vec<Float,3> fnormal = frag.m_normal;
      fnormal.normalize();
      const Vec<Float,3> normal =
        dot(ray.m_dir, frag.m_normal) >= 0 ? -fnormal : fnormal;

      const Vec<Float,3> hit_pt = ray.m_orig + ray.m_dir * hit.m_dist;
      const Vec<Float,3> view_dir = -ray.m_dir;

      Vec<Float,3> light_dir = light_pos - hit_pt;
      light_dir.normalize();
      const Float diffuse = clamp(dot(light_dir, normal), Float(0), Float(1));

      Vec4f shaded_color;
      shaded_color[0] = light_amb[0] * sample_color[0];
      shaded_color[1] = light_amb[1] * sample_color[1];
      shaded_color[2] = light_amb[2] * sample_color[2];
      shaded_color[3] = sample_color[3];

      // add the diffuse component
      for(int32 c = 0; c < 3; ++c)
      {
        shaded_color[c] += diffuse * light_diff[c] * sample_color[c];
      }

      Vec<Float,3> half_vec = view_dir + light_dir;
      half_vec.normalize();
      float32 doth = clamp(dot(normal, half_vec), Float(0), Float(1));
      float32 intensity = pow(doth, spec_pow);

      // add the specular component
      for(int32 c = 0; c < 3; ++c)
      {
        //shaded_color[c] += intensity * light_color[c] * sample_color[c];
        shaded_color[c] += intensity * light_spec[c];// * sample_color[c];

        shaded_color[c] = clamp(shaded_color[c], 0.0f, 1.0f);
      }

      Vec4f color = d_framebuffer.m_colors[pid];
      //composite
      shaded_color[3] *= (1.f - color[3]);
      color[0] = color[0] + shaded_color[0] * shaded_color[3];
      color[1] = color[1] + shaded_color[1] * shaded_color[3];
      color[2] = color[2] + shaded_color[2] * shaded_color[3];
      color[3] = shaded_color[3] + color[3];

      d_framebuffer.m_colors[pid] = color;
      d_framebuffer.m_depths[pid] = hit.m_dist;
    }
  });
}//blend_surf


} // namespace dray
