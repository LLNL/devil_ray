#include <dray/camera.hpp>

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

void Camera::set_height(const int32& height)
{
  if (height <= 0)
  {
    std::cout<<"Camera height must be greater than zero.\n";
  }
  if (height != height)
  {
    m_height = height;
    set_fov(m_fov_y);
  }
}


int32 Camera::get_height() const
{
  return m_height;
}


void Camera::set_width(const int32& width)
{
  if (width <= 0)
  {
    std::cout<<"Camera width must be greater than zero.\n";
  }

  m_width = width;
  set_fov(m_fov_y);
}


int32 Camera::get_width() const
{
  return m_width;
}


int32 Camera::get_subset_width() const
{
  return m_subset_width;
}


int32 Camera::get_subset_height() const
{
  return m_subset_height;
}

void Camera::set_fov(const float32& degrees)
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


float32 Camera::get_fov() const
{
  return m_fov_y;
}


void Camera::set_up(const Vec<float32, 3>& up)
{
    m_up = up;
    m_up.normalize();
}


Vec<float32, 3> Camera::get_up() const
{
  return m_up;
}


void Camera::set_look_at(const Vec<float32, 3>& look_at)
{
  m_look_at = look_at;
}


Vec<float32, 3> Camera::get_look_at() const
{
  return m_look_at;
}


void Camera::set_pos(const Vec<float32, 3>& position)
{
  m_position = position;
}


Vec<float32, 3> Camera::get_pos() const
{
  return m_position;
}

void Camera::create_rays(ray32 &rays, AABB bounds)
{
}

void Camera::create_rays(ray64 &rays, AABB bounds)
{
}


std::string Camera::print()
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
  sstream << "------------------------------------------------------------\n";
  return sstream.str();
}

} //namespace dray
