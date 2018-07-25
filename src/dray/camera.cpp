#include <dray/camera.hpp>

#include <dray/array_utils.hpp>
#include <dray/policies.hpp>

#include <sstream>
namespace dray
{


Camera::Camera()
{
  m_height = 500;
  m_width = 500;
  m_subset_width = 500;
  m_subset_height = 500;
  m_subset_min_x = 0;
  m_subset_min_y = 0;
  m_fov_y = 30.f;
  m_fov_x = 30.f;
  m_look[0] = 0.f;
  m_look[1] = 0.f;
  m_look[2] = -1.f;
  m_look_at[0] = 0.f;
  m_look_at[1] = 0.f;
  m_look_at[2] = -1.f;
  m_up[0] = 0.f;
  m_up[1] = 1.f;
  m_up[2] = 0.f;
  m_position[0] = 0.f;
  m_position[1] = 0.f;
  m_position[2] = 0.f;
}

Camera::~Camera()
{
}

void 
Camera::set_height(const int32& height)
{
  if (height <= 0)
  {
    std::cout<<"Camera height must be greater than zero.\n";
  }
  if (m_height != height)
  {
    m_height = height;
    m_subset_height = height;
    set_fov(m_fov_y);
  }
}

int32 
Camera::get_height() const
{
  return m_height;
}


void 
Camera::set_width(const int32& width)
{
  if (width <= 0)
  {
    std::cout<<"Camera width must be greater than zero.\n";
  }

  m_width = width;
  m_subset_width = width;
  set_fov(m_fov_y);
}


int32 
Camera::get_width() const
{
  return m_width;
}


int32 
Camera::get_subset_width() const
{
  return m_subset_width;
}


int32 
Camera::get_subset_height() const
{
  return m_subset_height;
}

void 
Camera::set_fov(const float32& degrees)
{
  if (degrees <= 0)
  {
    std::cout<<"Camera feild of view must be greater than zero.\n";
  }
  if (degrees > 180)
  {
    std::cout<<"Camera feild of view must be less than 180.\n";
  }

  float32 new_fov_y = degrees;
  float32 new_fov_x;

  float32 fov_y_rad = (new_fov_y * static_cast<float32>(pi())) / 180.0f;

  // Use the tan function to find the distance from the center of the image to the top (or
  // bottom). (Actually, we are finding the ratio of this distance to the near plane distance,
  // but since we scale everything by the near plane distance, we can use this ratio as a scaled
  // proxy of the distances we need.)
  float32 vertical_distance = tan(0.5f * fov_y_rad);

  // Scale the vertical distance by the aspect ratio to get the horizontal distance.
  float32 aspect = float32(m_width) / float32(m_height);
  float32 horizontal_distance = aspect * vertical_distance;

  // Now use the arctan function to get the proper field of view in the x direction.
  float32 fov_x_rad = 2.0f * atan(horizontal_distance);
  new_fov_x = 180.0f * (fov_x_rad / static_cast<float32>(pi()));
  m_fov_x = new_fov_x;
  m_fov_y = new_fov_y;
}


float32 
Camera::get_fov() const
{
  return m_fov_y;
}


void 
Camera::set_up(const Vec<float32, 3>& up)
{
    m_up = up;
    m_up.normalize();
}


Vec<float32, 3> 
Camera::get_up() const
{
  return m_up;
}


void 
Camera::set_look_at(const Vec<float32, 3>& look_at)
{
  m_look_at = look_at;
}


Vec<float32, 3> 
Camera::get_look_at() const
{
  return m_look_at;
}


void 
Camera::set_pos(const Vec<float32, 3>& position)
{
  m_position = position;
}


Vec<float32, 3> 
Camera::get_pos() const
{
  return m_position;
}

void 
Camera::create_rays(ray32 &rays, AABB bounds)
{
  create_rays_imp(rays, bounds);
}

void 
Camera::create_rays(ray64 &rays, AABB bounds)
{
  create_rays_imp(rays, bounds);
}

template<typename T>
void 
Camera::create_rays_imp(Ray<T> &rays, AABB bounds)
{
  int32 num_rays = m_width * m_height;
  // TODO: find subset
  // for now just set 
  m_subset_width = m_width;
  m_subset_height = m_height;
  m_subset_min_x = 0;
  m_subset_min_y = 0;

  rays.resize(num_rays);

  Vec<T,3> pos;
  pos[0] = m_position[0];
  pos[1] = m_position[1];
  pos[2] = m_position[2];

  m_look = m_look_at - m_position;

  array_memset_vec(rays.m_orig, pos);
  array_memset(rays.m_near, T(0.f));
  array_memset(rays.m_far, infinity<T>());

  //TODO Why don't we set rays.m_dist to the same 0.0 as m_near?
   
  gen_perspective(rays);

  rays.m_active_rays = array_counting(rays.size(),0,1);
}



std::string 
Camera::print()
{
  std::stringstream sstream;
  sstream << "------------------------------------------------------------\n";
  sstream << "Position : [" << m_position[0] << ",";
  sstream << m_position[1] << ",";
  sstream << m_position[2] << "]\n";
  sstream << "LookAt   : [" << m_look_at[0] << ",";
  sstream << m_look_at[1] << ",";
  sstream << m_look_at[2] << "]\n";
  sstream << "FOV_X    : " << m_fov_x<< "\n";
  sstream << "Up       : [" << m_up[0] << ",";
  sstream << m_up[1] << ",";
  sstream << m_up[2] << "]\n";
  sstream << "Width    : " << m_width << "\n";
  sstream << "Height   : " << m_height << "\n";
  sstream << "Subset W : " << m_subset_width << "\n";
  sstream << "Subset H : " << m_subset_height << "\n";
  sstream << "------------------------------------------------------------\n";
  return sstream.str();
}


template<typename T>
void 
Camera::gen_perspective(Ray<T> &rays)
{
  Vec<T, 3> nlook;
  Vec<T, 3> delta_x;
  Vec<T, 3> delta_y;

  T thx = tanf((m_fov_x * T(pi()) / 180.f) * .5f);
  T thy = tanf((m_fov_y * T(pi()) / 180.f) * .5f);
  Vec<float32, 3> ruf = cross(m_look, m_up);
  Vec<T, 3> ru;
  ru[0] = ruf[0];
  ru[1] = ruf[1];
  ru[2] = ruf[2];

  ru.normalize();

  Vec<float32, 3> rvf = cross(ruf, m_look);
  Vec<T, 3> rv;
  rv[0] = rvf[0];
  rv[1] = rvf[1];
  rv[2] = rvf[2];

  rv.normalize();
  delta_x = ru * (2 * thx / (T)m_width);
  delta_y = rv * (2 * thy / (T)m_height);

  nlook[0] = m_look[0];
  nlook[1] = m_look[1];
  nlook[2] = m_look[2];
  nlook.normalize();
  
  const int size = rays.size();

  Vec<T, 3> *dir_ptr = rays.m_dir.get_device_ptr();
  int32 *pid_ptr = rays.m_pixel_id.get_device_ptr();
  // something weird is happening with the 
  // lambda capture
  const int32 w = m_width;
  const int32 sub_min_x = m_subset_min_x;
  const int32 sub_min_y = m_subset_min_y;
  const int32 sub_w = m_subset_width;
  RAJA::forall<for_policy>(RAJA::RangeSegment(0, size), [=] DRAY_LAMBDA (int32 idx)
  {

    int32 i = int32(idx) % sub_w;
    int32 j = int32(idx) / sub_w;
    i += sub_min_x;
    j += sub_min_y;
    // Write out the global pixelId
    pid_ptr[idx] = static_cast<int32>(j * w + i);
    Vec<T, 3> ray_dir = nlook + delta_x * ((2.f * T(i) - T(w)) / 2.0f) +
      delta_y * ((2.f * T(j) - T(w)) / 2.0f);
    // avoid some numerical issues
    for (int32 d = 0; d < 3; ++d)
    {
      if (ray_dir[d] == 0.f)
        ray_dir[d] += 0.0000001f;
    }

    ray_dir.normalize();
    
    dir_ptr[idx] = ray_dir;
    //printf("Ray dir %f %f %f\n", ray_dir[0], ray_dir[1], ray_dir[2]);
  });

}

void 
Camera::reset_to_bounds(const AABB bounds,
                        const float64 xpad,
                        const float64 ypad,
                        const float64 zpad)
{
  AABB db;

  float64 pad = xpad * (bounds.m_x.max() - bounds.m_x.min());
  db.m_x.include(bounds.m_x.max() + pad);
  db.m_x.include(bounds.m_x.min() - pad);

  pad = ypad * (bounds.m_y.max() - bounds.m_y.min());
  db.m_y.include(bounds.m_y.max() + pad);
  db.m_y.include(bounds.m_y.min() - pad);

  pad = zpad * (bounds.m_z.max() - bounds.m_z.min());
  db.m_z.include(bounds.m_z.max() + pad);
  db.m_z.include(bounds.m_z.min() - pad);

  Vec3f proj_dir = m_position - m_look_at;
  proj_dir.normalize();

  Vec3f center = db.center();
  m_look_at = center;

  Vec3f extent;
  extent[0] = float32(db.m_x.length());
  extent[1] = float32(db.m_y.length());
  extent[2] = float32(db.m_z.length());
  float32 diagonal = extent.magnitude();
  m_position = center + proj_dir * diagonal* 1.0f;
  set_fov(60.0f);

}
} //namespace dray
