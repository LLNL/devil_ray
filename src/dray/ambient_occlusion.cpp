
#include <dray/ambient_occlusion.hpp>
#include <dray/array_utils.hpp>
#include <dray/policies.hpp>
#include <dray/math.hpp>

//#include <sstream>
namespace dray
{

template<typename T>
float32 AmbientOcclusion<T>::nudge_dist = 0.000001f;


template<typename T>
template<typename C>  // for color factor.
void
//AmbientOcclusion<T>::calc_occlusion(
//    Array<Vec<T,3>> &hit_points,
//    Array<Vec<T,3>> &surface_normals,
//    int32 occ_samples,
//    Array<C> &occ_component)
AmbientOcclusion<T>::calc_occlusion(
    Ray<T> &incoming_rays,
    int32 occ_samples,          // assume gathered? Here we don't need rays that didn't intersect anything.
    Array<C> &occ_component)
{
  // - Should we use some kind of indexing? Assume that some normals might have the same identity?
  // - Do Halton2D over ALL ao-samples at once... want to use large sample idxs.
  //int32 num_incoming_hits = hit_points.size();
  //assert(num_incoming_hits == surface_normals.size());

  int32 num_incoming_rays = incoming_rays.size();
  //assert(num_incoming_rays == occ_component.size());
  int32 total_occ_samples = occ_samples * num_incoming_rays;

  // Get read-only device pointers to fields of incoming rays.
  const Vec<T,3> *in_dir_ptr = incoming_rays.m_dir.get_device_ptr_const();
  const Vec<T,3> *in_orig_ptr = incoming_rays.m_orig.get_device_ptr_const();
  const T *in_dist_ptr = incoming_rays.m_dist.get_device_ptr_const();
  const int32 *in_hit_idx_ptr = incoming_rays.m_hit_idx.get_device_ptr_const();

  // Allocate vector arrays for the normal and tangent spaces.  //TODO These arrays will auto destruct, right?
  Array<Vec<T,3>> tangent_x;
  Array<Vec<T,3>> tangent_y;
  Array<Vec<T,3>> normal;
  tangent_x.resize(num_incoming_rays);
  tangent_y.resize(num_incoming_rays);
  normal.resize(num_incoming_rays);
  Vec<T,3> *tangent_x_ptr = tangent_x.get_device_ptr();
  Vec<T,3> *tangent_y_ptr = tangent_y.get_device_ptr();
  Vec<T,3> *normal_ptr = normal.get_device_ptr();

  // For each incoming hit, get normal and construct basis for tangent space.
  // n.b.: We need to do this for each hit, not just for each intersected element.
  //   Unless the elements are flat (triangles), the surface normal can vary
  //   within a single element, depending on the location of the hit point.
  RAJA::forall<for_policy>(RAJA::RangeSegment(0, num_incoming_rays), [=] DRAY_LAMBDA (int32 in_ray_idx)
  {
    //normal_ptr[in_ray_idx] =...  //TODO calculate normal... surely for triangles there is a method for this already.
    ConstructTangentBasis(normal_ptr[in_ray_idx], tangent_x_ptr[in_ray_idx], tangent_y_ptr[in_ray_idx]);
    
    //TODO ? Should we scatter the results at this point?
    // Is it better to avoid misaligned memory, or to save the repeated computations?
  });

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, total_occ_samples), [=] DRAY_LAMBDA (int32 occ_sample_idx)
  {
      // Need the index of the incoming ray and corresponding tangent space.
      int32 in_ray_idx = occ_sample_idx % occ_samples;

      // Get Halton hemisphere sample in local coordinates.
      Vec<T,3> occ_local_direction = CosineWeightedHemisphere(occ_sample_idx);

      // Map these coordinates onto the local frame, get world coordinates.
      Vec<T,3> occ_direction;
      occ_direction[0] = dot(occ_local_direction, tangent_x_ptr[in_ray_idx]);
      occ_direction[1] = dot(occ_local_direction, tangent_y_ptr[in_ray_idx]);
      occ_direction[2] = dot(occ_local_direction, normal_ptr[in_ray_idx]);

      //TODO Construct new occ_ray.

      //TODO Perform intersection test for the new ray.
      //TODO Export result of the intersection test.
  });

  //TODO Some RAJA invocation that will
  // 1. Count the number of occlusion hits per incoming ray.
  // 2. Scale the sum -> [0.0, 1.0].


}

//template<typename T>
//void AmbientOcclusion<T>::gen_occlusion(
//    Vec<T,3> &hit_pt,
//    Vec<T,3> &normal,
//    //int32 occ_samples,    //Assume that occ_samples == occ_rays.size()
//    Ray<T> &occ_rays)
//{
//
//  occ_samples = occ_rays.size();
//  RAJA::forall<for_policy>(RAJA::RangeSegment(0, occ_samples), [=] DRAY_LAMBDA (int32 idx)
//  {
//  }
//
//  //Array<Vec<T,3>> m_dir;
//  //Array<Vec<T,3>> m_orig;
//  occ_rays.
//
//  Array<Vec<T,3>> directions = 
//
//}



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
template void AmbientOcclusion<float32>::calc_occlusion(
    Ray<float32> &, int32, Array<float32> &);

} //namespace dray
