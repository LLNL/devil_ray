
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
const T AmbientOcclusion<T>::nudge_dist = 0.00005f;


template<typename T>
Ray<T> AmbientOcclusion<T>::gen_occlusion(
    const IntersectionContext<T> intersection_ctx,
    const int32 occ_samples,
    const T occ_near,
    const T occ_far)
{
  ManagedArray<int32> unused_array;
  return AmbientOcclusion<T>::gen_occlusion(
      intersection_ctx, occ_samples, occ_near, occ_far, unused_array);
}

template<typename T>
Ray<T> AmbientOcclusion<T>::gen_occlusion(
    const IntersectionContext<T> intersection_ctx,
    const int32 occ_samples,
    const T occ_near,
    const T occ_far,
    ManagedArray<int32> &compact_indexing)
{
  // Some intersection contexts may represent non-intersections.
  // We only produce occlusion rays for valid intersections.
  // Therefore we re-index the set of rays which actually hit something.
  //   0 .. ray_idx .. (num_prim_rays-1)
  //   0 .. hit_idx .. (num_prim_hits-1)
  int32 num_prim_rays = intersection_ctx.size();
  int32 num_prim_hits;
  compact_indexing = array_compact_indices(intersection_ctx.m_is_valid, num_prim_hits);

  // Initialize entropy array, needed before sampling Halton hemisphere.
  ManagedArray<int32> entropy = array_random(num_prim_hits, time(NULL), num_prim_hits);  //TODO choose right upper bound

  // Allocate new occlusion rays.
  Ray<T> occ_rays;
  occ_rays.resize(num_prim_hits * occ_samples);

  // The near and far fields are the same for all occlusion rays.
  array_memset(occ_rays.m_near, occ_near);
  array_memset(occ_rays.m_far, occ_far);

  // "l" == "local": Capture parameters to local variables, for loop kernel.
  const T     l_nudge_dist = AmbientOcclusion<T>::nudge_dist;
  const int32 l_occ_samples = occ_samples;

  // Input pointers.
  const int32 *entropy_ptr = entropy.get_device_ptr_const();

  const int32 *is_valid_ptr  = intersection_ctx.m_is_valid.get_device_ptr_const();
  const Vec<T,3> *hit_pt_ptr = intersection_ctx.m_hit_pt.get_device_ptr_const();
  const Vec<T,3> *normal_ptr = intersection_ctx.m_normal.get_device_ptr_const();
  const Vec<T,3> *ray_dir_ptr = intersection_ctx.m_ray_dir.get_device_ptr_const();
  const int32 *pixel_id_ptr  = intersection_ctx.m_pixel_id.get_device_ptr_const();

  const int32 *compact_indexing_ptr = compact_indexing.get_device_ptr_const();

  // Output pointers.
  Vec<T,3> *occ_dir_ptr = occ_rays.m_dir.get_device_ptr();
  Vec<T,3> *occ_orig_ptr = occ_rays.m_orig.get_device_ptr();
  int32 *occ_pixel_id_ptr = occ_rays.m_pixel_id.get_device_ptr();

  // For each incoming hit, generate (occ_samples) new occlusion rays.
  RAJA::forall<for_policy>(RAJA::RangeSegment(0, num_prim_rays * occ_samples), [=] DRAY_LAMBDA (int32 ii)
  {
    // We launch (occ_samples) instances for each incoming ray.
    // This thread is identified by two indices:
    //  0 <= prim_ray_idx   < num_prim_rays
    //  0 <= occ_sample_idx < occ_samples
    int32 prim_ray_idx = ii / l_occ_samples;
    int32 sample = ii % l_occ_samples;

    // First test whether the intersection is valid; only proceed if it is.
    if (is_valid_ptr[prim_ray_idx])
    {
      // Get normal and construct basis for tangent space.
      // Note: We need to do this for each hit, not just for each intersected element.
      //   Unless the elements are flat (triangles), the surface normal can vary
      //   within a single element, depending on the location of the hit point.
      Vec<T,3> normal = normal_ptr[prim_ray_idx],  tangent_x, tangent_y;
      ConstructTangentBasis(normal, tangent_x, tangent_y);

      // Make a 'nudge vector' to displace occlusion rays, avoid self-intersection.
      /// Vec<T,3> nudge = normal * l_nudge_dist;
      Vec<T,3> nudge = ray_dir_ptr[prim_ray_idx] * (-l_nudge_dist);

      // Find output indices for this sample.
      int32 prim_hit_idx = compact_indexing_ptr[prim_ray_idx];
      int32 occ_offset = prim_hit_idx * l_occ_samples;

      // Get Halton hemisphere sample in local coordinates.
      Vec<T,3> occ_local_direction =
          CosineWeightedHemisphere(entropy_ptr[prim_hit_idx] + sample);

      // Map these coordinates onto the local frame, get world coordinates.
      Vec<T,3> occ_direction =
          tangent_x * occ_local_direction[0] +
          tangent_y * occ_local_direction[1] +
          normal    * occ_local_direction[2];
      occ_direction.normalize();

      // Initialize new occ_ray.
      occ_dir_ptr[occ_offset + sample] = occ_direction;
      occ_orig_ptr[occ_offset + sample] = hit_pt_ptr[prim_ray_idx] + nudge;
      occ_pixel_id_ptr[occ_offset + sample] = pixel_id_ptr[prim_ray_idx];
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
