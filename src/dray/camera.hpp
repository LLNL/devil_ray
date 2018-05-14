#ifndef DRAY_CAMERA_HPP
#define DRAY_CAMERA_HPP

#include <dray/aabb.hpp>
#include <dray/types.hpp>
#include <dray/ray.hpp>
#include <dray/vec.hpp>

namespace dray
{

class Camera
{

private:

  int32 m_height;
  int32 m_width;
  int32 m_subset_width;
  int32 m_subset_height;
  int32 m_subset_min_x;
  int32 m_subset_min_y;
  float32 m_fov_x;
  float32 m_fov_y;

  Vec<float32, 3> m_look;
  Vec<float32, 3> m_up;
  Vec<float32, 3> m_look_at;
  Vec<float32, 3> m_position;

  template<typename T>
  void create_rays_imp(Ray<T> &rays, AABB bounds);

public:
  Camera();

  ~Camera();

  std::string print();

  void reset_to_bounds(const AABB bounds,
                       const float64 xpad = 0.,
                       const float64 ypad = 0.,
                       const float64 zpad = 0.);

  void set_height(const int32 &height);

  int32 get_height() const;

  void set_width(const int32 &width);

  int32 get_width() const;

  int32 get_subset_width() const;

  int32 get_subset_height() const;

  void set_fov(const float32 &degrees);

  float32 get_fov() const;

  void set_up(const Vec<float32, 3> &up);

  void set_pos(const Vec<float32, 3> &position);

  Vec<float32, 3> get_pos() const;

  Vec<float32, 3> get_up() const;

  void set_look_at(const Vec<float32, 3> &look_at);

  Vec<float32, 3> get_look_at() const;

  void create_rays(ray32 &rays, AABB bounds = AABB());

  void create_rays(ray64 &rays, AABB bounds = AABB());


  template<typename T>
  void gen_perspective(Ray<T> &rays);
}; // class camera

} // namespace dray
#endif
