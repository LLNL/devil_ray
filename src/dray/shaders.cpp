#include <dray/shaders.hpp>

#include <dray/math.hpp>
#include <dray/policies.hpp>

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
                     dray::Vec<float, 4> &bg_color)
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
Shader::set_color_table(ColorTable &color_table) 
{
  //color_table.sample(m_color_samples, m_color_map);
  m_color_table = color_table;
  std::cout<<"Setting color table *******\n"; 
} // set_color table 

template<typename T>
void Shader::blend(Array<Vec4f> &color_buffer,
                   ShadingContext<T> &shading_ctx)

{
  Array<Vec4f> color_map;
  m_color_table.sample(m_color_samples, color_map);

  if(color_map.size() == 0)
  {
    // set up a default color table
    ColorTable color_table("cool2warm");
    m_color_table = color_table;
    m_color_table.sample(m_color_samples, color_map);
  }

  const int32 *pid_ptr = shading_ctx.m_pixel_id.get_device_ptr_const();
  const int32 *is_valid_ptr = shading_ctx.m_is_valid.get_device_ptr_const();
  const T *sample_val_ptr = shading_ctx.m_sample_val.get_device_ptr_const();

  const Vec<T,3> *normal_ptr = shading_ctx.m_normal.get_device_ptr_const();
  const Vec<T,3> *hit_pt_ptr = shading_ctx.m_hit_pt.get_device_ptr_const();
  const Vec<T,3> *ray_dir_ptr = shading_ctx.m_ray_dir.get_device_ptr_const();

  const Vec4f *color_map_ptr = color_map.get_device_ptr_const();

  Vec4f *img_ptr = color_buffer.get_device_ptr();

  const int color_map_size = color_map.size();

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, shading_ctx.size()), [=] DRAY_LAMBDA (int32 ii)
  {
    if (is_valid_ptr[ii])
    {
      int32 pid = pid_ptr[ii];
      const T sample_val = sample_val_ptr[ii];
      int32 sample_idx = static_cast<int32>(sample_val * float32(color_map_size - 1));

      Vec4f sample_color = color_map_ptr[sample_idx];
      //std::cout<<"sample color "<<sample_color<<" "<<sample_val<<"\n";
///      Vec<T,3> normal = normal_ptr[ii];
///      Vec<T,3> hit_pt = hit_pt_ptr[ii];
///      Vec<T,3> view_dir = -ray_dir_ptr[ii];
///      
///      Vec<T,3> light_dir = light_pos - hit_pt;
///      light_dir.normalize();
///      T diffuse = clamp(dot(light_dir, normal), T(0), T(1));
///
///      Vec4f shaded_color;
///      shaded_color[0] = light_amb[0];
///      shaded_color[1] = light_amb[1];
///      shaded_color[2] = light_amb[2];
///      shaded_color[3] = sample_color[3];
///      
///      // add the diffuse component
///      for(int32 c = 0; c < 3; ++c)
///      {
///        shaded_color[c] += diffuse * light_color[c] * sample_color[c];
///      }
///
///      Vec<T,3> half_vec = view_dir + light_dir;
///      half_vec.normalize();
///      float32 doth = clamp(dot(normal, half_vec), T(0), T(1));
///      float32 intensity = pow(doth, spec_pow);
///
///      // add the specular component
///      for(int32 c = 0; c < 3; ++c)
///      {
///        shaded_color[c] += intensity * light_color[c] * sample_color[c];
///      }

      Vec4f color = img_ptr[pid];
      //composite
      sample_color[3] *= (1.f - color[3]);
      color[0] = color[0] + sample_color[0] * sample_color[3];
      color[1] = color[1] + sample_color[1] * sample_color[3];
      color[2] = color[2] + sample_color[2] * sample_color[3];
      color[3] = sample_color[3] + color[3];
      img_ptr[pid] = color;
    }
  });
}//blend


template<typename T>
void Shader::blend_surf(Array<Vec4f> &color_buffer,
                   ShadingContext<T> &shading_ctx)

{
  printf("Shader::blend_surf()\n");

  Array<Vec4f> color_map;
  m_color_table.sample(m_color_samples, color_map);

  if(color_map.size() == 0)
  {
    // set up a default color table
    ColorTable color_table("cool2warm");
    m_color_table = color_table;
    m_color_table.sample(m_color_samples, color_map);
  }

  const int32 *pid_ptr = shading_ctx.m_pixel_id.get_device_ptr_const();
  const int32 *is_valid_ptr = shading_ctx.m_is_valid.get_device_ptr_const();
  const T *sample_val_ptr = shading_ctx.m_sample_val.get_device_ptr_const();

  const Vec<T,3> *normal_ptr = shading_ctx.m_normal.get_device_ptr_const();
  const Vec<T,3> *hit_pt_ptr = shading_ctx.m_hit_pt.get_device_ptr_const();
  const Vec<T,3> *ray_dir_ptr = shading_ctx.m_ray_dir.get_device_ptr_const();

  const Vec4f *color_map_ptr = color_map.get_device_ptr_const();

  Vec4f *img_ptr = color_buffer.get_device_ptr();

  const int color_map_size = color_map.size();

  const Vec<T,3> light_pos = {m_light.m_pos[0], m_light.m_pos[1], m_light.m_pos[2]};
    // Local for lambda.
  const Vec<float32,3> &light_amb = m_light.m_amb;
  const Vec<float32,3> &light_diff = m_light.m_diff;
  const Vec<float32,3> &light_spec = m_light.m_spec;
  const float32 &spec_pow = m_light.m_spec_pow; //shiny

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, shading_ctx.size()), [=] DRAY_LAMBDA (int32 ii)
  {
    if (is_valid_ptr[ii])
    {
      int32 pid = pid_ptr[ii];
      const T sample_val = sample_val_ptr[ii];
      int32 sample_idx = static_cast<int32>(sample_val * float32(color_map_size - 1));

      Vec4f sample_color = color_map_ptr[sample_idx];

      Vec<T,3> normal = normal_ptr[ii];
      Vec<T,3> hit_pt = hit_pt_ptr[ii];
      Vec<T,3> view_dir = -ray_dir_ptr[ii];
      
      Vec<T,3> light_dir = light_pos - hit_pt;
      light_dir.normalize();
      T diffuse = clamp(dot(light_dir, normal), T(0), T(1));

      Vec4f shaded_color;
      shaded_color[0] = light_amb[0];
      shaded_color[1] = light_amb[1];
      shaded_color[2] = light_amb[2];
      shaded_color[3] = sample_color[3];
      
      // add the diffuse component
      for(int32 c = 0; c < 3; ++c)
      {
        //shaded_color[c] += diffuse * light_color[c] * sample_color[c];
        shaded_color[c] += diffuse * light_diff[c] * sample_color[c];
      }

      Vec<T,3> half_vec = view_dir + light_dir;
      half_vec.normalize();
      float32 doth = clamp(dot(normal, half_vec), T(0), T(1));
      float32 intensity = pow(doth, spec_pow);

      // add the specular component
      for(int32 c = 0; c < 3; ++c)
      {
        //shaded_color[c] += intensity * light_color[c] * sample_color[c];
        shaded_color[c] += intensity * light_spec[c];// * sample_color[c];

        shaded_color[c] = clamp(shaded_color[c], 0.0f, 1.0f);
      }

      Vec4f color = img_ptr[pid];
      //composite
      shaded_color[3] *= (1.f - color[3]);
      color[0] = color[0] + shaded_color[0] * shaded_color[3];
      color[1] = color[1] + shaded_color[1] * shaded_color[3];
      color[2] = color[2] + shaded_color[2] * shaded_color[3];
      color[3] = shaded_color[3] + color[3];
      img_ptr[pid] = color;
    }
  });
}//blend_surf






template void  Shader::blend(Array<Vec4f> &color_buffer,
                             ShadingContext<float32> &shading_ctx);

template void  Shader::blend(Array<Vec4f> &color_buffer,
                             ShadingContext<float64> &shading_ctx);

template void  Shader::blend_surf(Array<Vec4f> &color_buffer,
                             ShadingContext<float32> &shading_ctx);

template void  Shader::blend_surf(Array<Vec4f> &color_buffer,
                             ShadingContext<float64> &shading_ctx);
} // namespace dray
