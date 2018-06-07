
#include <dray/ambient_occlusion.hpp>
#include <dray/intersection_context.hpp>
#include <dray/array_utils.hpp>
#include <dray/policies.hpp>
#include <dray/math.hpp>

#include <stdio.h>  /* NULL */
#include <time.h>   /* time */

//#include <sstream>
namespace dray
{

template<typename T>
//const T AmbientOcclusion<T>::nudge_dist = 0.000001f;
const T AmbientOcclusion<T>::nudge_dist = 0.0001f;


template<typename T>
Ray<T> AmbientOcclusion<T>::gen_occlusion(
    IntersectionContext<T> intersection_ctx,
    int32 occ_samples,
    T occ_near,
    T occ_far)
{
  Array<int32> unused_array;
  return AmbientOcclusion<T>::gen_occlusion(
      intersection_ctx, occ_samples, occ_near, occ_far, unused_array);
}

template<typename T>
Ray<T> AmbientOcclusion<T>::gen_occlusion(
    IntersectionContext<T> intersection_ctx,
    int32 occ_samples,
    T occ_near,
    T occ_far,
    Array<int32> &compact_indexing)
{
  // The arrays in intersection_ctx may be non-yet-compacted: Some rays do not hit.
  // But the returned arrays need to be compacted.
  //
  // Therefore we have num_incoming_rays >= num_incoming_hits.
  // We return (num_incoming_hits * occ_samples) occlusion rays.

  int32 num_incoming_rays = intersection_ctx.size();
  int32 num_incoming_hits;
  Array<int32> hit_valid_idx = array_compact_indices(intersection_ctx.m_is_valid, num_incoming_hits);

  // Output hit_valid_idx as compact_indexing.  //TODO: Simply rename hit_valild_idx as compact_indexing.
  compact_indexing = hit_valid_idx;

  // Read-only pointer to hit_valid_idx.
  const int32 *hit_valid_idx_ptr = hit_valid_idx.get_device_ptr_const();

  const int32 *is_valid_device_ptr = intersection_ctx.m_is_valid.get_device_ptr_const();
  const Vec<T,3> *hit_pt_ptr = intersection_ctx.m_hit_pt.get_device_ptr_const();
  const Vec<T,3> *normal_ptr = intersection_ctx.m_normal.get_device_ptr_const();
  const int32 *pixel_id_ptr  = intersection_ctx.m_pixel_id.get_device_ptr_const();

  // Initialize entropy array, needed before sampling Halton hemisphere.
  Array<int32> entropy_array = array_random(num_incoming_hits, time(NULL), num_incoming_hits);  //TODO choose right upper bound
  //Array<int32> entropy_array = array_random(num_incoming_hits, 999, num_incoming_hits);  //TODO seed using time(), as above.
  const int32 *entropy_array_ptr = entropy_array.get_device_ptr_const();

  // Allocate new occlusion rays.
  Ray<T> occ_rays;
  occ_rays.resize(num_incoming_hits * occ_samples);
  Vec<T,3> *occ_dir_ptr = occ_rays.m_dir.get_device_ptr();
  Vec<T,3> *occ_orig_ptr = occ_rays.m_orig.get_device_ptr();
  int32 *occ_pixel_id_ptr = occ_rays.m_pixel_id.get_device_ptr();

  // The near and far fields are uniform, so initialize them now.
  array_memset(occ_rays.m_near, occ_near);
  array_memset(occ_rays.m_far, occ_far);

  // "l" == "local": Capture parameters to local variables, for device.
  const T l_nudge_dist = AmbientOcclusion<T>::nudge_dist;
  const int32 l_occ_samples = occ_samples;

  // For each incoming hit, generate (occ_samples) new occlusion rays.
  // Need to initialize origin, direction, and pixel_id.
  RAJA::forall<for_policy>(RAJA::RangeSegment(0, num_incoming_rays * occ_samples), [=] DRAY_LAMBDA (int32 ii)
  {
    // We launch (occ_samples) instances for each incoming ray.
    // This thread is identified by two indices:
    //  0 <= in_ray_idx     < num_incoming_rays
    //  0 <= occ_sample_idx < occ_samples
    int32 in_ray_idx = ii / l_occ_samples;
    int32 occ_sample_idx = ii % l_occ_samples;

    // First test whether the intersection is valid; only proceed if it is.
    if (is_valid_device_ptr[in_ray_idx])
    {
      // Get normal and construct basis for tangent space.
      // Note: We need to do this for each hit, not just for each intersected element.
      //   Unless the elements are flat (triangles), the surface normal can vary
      //   within a single element, depending on the location of the hit point.
      Vec<T,3> normal = normal_ptr[in_ray_idx];
      Vec<T,3> tangent_x;
      Vec<T,3> tangent_y;
      ConstructTangentBasis(normal, tangent_x, tangent_y);

      // Make a 'nudge vector' to displace occlusion rays off the surface.
      Vec<T,3> nudge = normal * l_nudge_dist;

      // Find output indices for this sample.
      int32 hit_valid_idx_here = hit_valid_idx_ptr[in_ray_idx];
      int32 occ_offset_hit = l_occ_samples * hit_valid_idx_here;

      // Get Halton hemisphere sample in local coordinates.
      Vec<T,3> occ_local_direction = CosineWeightedHemisphere(
          entropy_array_ptr[hit_valid_idx_here] + occ_sample_idx);

      // Map these coordinates onto the local frame, get world coordinates.
      Vec<T,3> occ_direction =
          tangent_x * occ_local_direction[0] +
          tangent_y * occ_local_direction[1] +
          normal * occ_local_direction[2];

      // Initialize new occ_ray.
      occ_dir_ptr[occ_offset_hit + occ_sample_idx] = occ_direction;
      occ_orig_ptr[occ_offset_hit + occ_sample_idx] = hit_pt_ptr[in_ray_idx] + nudge;
      occ_pixel_id_ptr[occ_offset_hit + occ_sample_idx] = pixel_id_ptr[in_ray_idx];
    }
  });

  return occ_rays;
}

// ----------------------------------------------

// These sampling methods were adapted from https://gitlab.kitware.com/mclarsen/vtk-m/blob/pathtracer/vtkm/rendering/raytracing/Sampler.h
// - Halton2D
// - CosineWeightedHemisphere
// - ConstructTangentBasis (factored from CosineWeightedHemisphere).
// TODO Convert camelCase (vtk-m) to lower_case (dray) ?

template <typename T>
template <int32 Base>
void AmbientOcclusion<T>::Halton2D(
    const int32 &sampleNum, 
    Vec<T,2> &coord)
{
  //generate base2 halton (use bit arithmetic)
  T x = 0.0f;
  T xadd = 1.0f;
  uint32 b2 = 1 + sampleNum;
  while (b2 != 0)
  {
    xadd *= 0.5f;
    if ((b2 & 1) != 0)
    x += xadd;
    b2 >>= 1;
  }

  //generate arbitrary Base Halton
  T y = 0.0f;
  T yadd = 1.0f;
  int32 bn = 1 + sampleNum;
  while (bn != 0)
  {
    yadd *= 1.0f / (T) Base;
    y += (T)(bn % Base) * yadd;
    bn /= Base;
  }

  coord[0] = x;
  coord[1] = y;
} // Halton2D

template <typename T>
Vec<T,3> AmbientOcclusion<T>::CosineWeightedHemisphere(const int32 &sampleNum)
{
  Vec<T,2> xy;
  Halton2D<3>(sampleNum,xy);
  const T r = sqrt(xy[0]);
  const T theta = 2 * pi() * xy[1];

  Vec<T,3> direction;
  direction[0] = r * cos(theta);
  direction[1] = r * sin(theta);
  direction[2] = sqrt(max(0.0f, 1.f - xy[0]));
  return direction;

  //Vec<T,3> sampleDir;
  //sampleDir[0] = dot(direction, xAxis);
  //sampleDir[1] = dot(direction, yAxis);
  //sampleDir[2] = dot(direction, normal);
  //return sampleDir;
}

template <typename T>
void AmbientOcclusion<T>::ConstructTangentBasis(
    const Vec<T,3> &normal,
    Vec<T,3> &xAxis,
    Vec<T,3> &yAxis)
{
  //generate orthoganal basis about normal (i.e. basis for tangent space).
  //kz will be the axis idx (0,1,2) most aligned with normal.
  //TODO MAI [2018-05-30] I propose we instead choose the axis LEAST aligned with normal;
  // this amounts to flipping all the > to instead be <.
  int32 kz = 0;
  if(fabs(normal[0]) > fabs(normal[1])) 
  {
    if(fabs(normal[0]) > fabs(normal[2]))
      kz = 0;
    else
      kz = 2;
  }
  else 
  {
    if(fabs(normal[1]) > fabs(normal[2]))
      kz = 1;
    else
      kz = 2;
  }
  //nonNormal will be the axis vector most aligned with normal. (future: least aligned?)
  Vec<T,3> notNormal;
  notNormal[0] = 0.f;
  notNormal[1] = 0.f;
  notNormal[2] = 0.f;
  notNormal[(kz+1)%3] = 1.f;   //[M.A.I. 5/31]

  xAxis = cross(normal, notNormal);
  xAxis.normalize();
  yAxis = cross(normal, xAxis);
  yAxis.normalize();
}



// Explicit template instantiations.

template class AmbientOcclusion<float32>;
//template void AmbientOcclusion<float32>::calc_occlusion(
//    Array<Vec<float32,3>> &, Array<Vec<float32,3>> &, int32, Array<float32> &);

//template void AmbientOcclusion<float32>::calc_occlusion(
//    Ray<float32> &, int32, Array<float32> &);

} //namespace dray
