#ifndef DRAY_RAY_HPP
#define DRAY_RAY_HPP

#include <dray/array.hpp>
#include <dray/exports.hpp>
#include <dray/types.hpp>
#include <dray/vec.hpp>
// TODO: includes that need to be moved to ray ops
#include <dray/aabb.hpp>

namespace dray
{

class Ray
{
public:
  Vec<Float,3> m_dir;
  Vec<Float,3> m_orig;
  Float        m_near;
  Float        m_far;
  Float        m_dist;
  int32        m_pixel_id;

  // TODO factor these out, since not intrinsic to a ray. For now just pretend they aren't members.
  int32        m_hit_idx;
  Vec<Float,3> m_hit_ref_pt;    // TODO have to fix triangle mesh and MFEM- Mesh/GridFunction before removing.
  int32        m_active;

  //static Ray gather_rays(const Ray rays, const Array<int32> indices);
};

static
std::ostream & operator << (std::ostream &out, const Ray &r)
{
  out<<r.m_pixel_id;
  return out;
}

Array<Vec<Float,3>> calc_tips(const Array<Ray> &rays);

Array<int32> active_indices(const Array<Ray> &rays);

void advance_ray(Array<Ray> &rays, float32 distance);

//
// utility function to find estimated intersection with the bounding
// box of the mesh
//
// After calling:
//   rays m_near   : set to estimated mesh entry
//   rays m_far    : set to estimated mesh exit
//
//   if ray missed then m_far <= m_near, if ray hit then m_far > m_near.
//
void calc_ray_start(Array<Ray> &rays, AABB<> bounds);

} // namespace dray
#endif
