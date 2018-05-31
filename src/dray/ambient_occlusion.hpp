#ifndef DRAY_AMBIENT_OCCLUSION_HPP
#define DRAY_AMBIENT_OCCLUSION_HPP

//#include <dray/aabb.hpp>
#include <dray/types.hpp>
#include <dray/ray.hpp>
#include <dray/vec.hpp>

namespace dray
{

template <typename T>
class AmbientOcclusion
{

public:

// Not sure where these functions should go...

  static float32 nudge_dist;

  //TODO update parameter description.
  /**
   * [in] hit_points
   * [in] surface_normals
   * [in] occ_samples
   * [out] occ_component
   */
  // TODO Ask if types should be adjusted.
  template<typename C>  // for color factor.
  static
  //void calc_occlusion(Array<Vec<T,3>> &hit_points, Array<Vec<T,3>> &surface_normals, int32 occ_samples, Array<C> &occ_component);
  void calc_occlusion(Ray<T> &incoming_rays, int32 occ_samples, Array<C> &occ_component);

  /**
   * [in] hit_pt
   * [in] normal
   * [in] occ_samples
   * [out] occ_rays
   */
  // TODO Likely this will be merged into the outer function, to avoid having separate Ray instantiations.
//  static
//  void gen_occlusion(Vec<T,3> &hit_pt, Vec<T,3> &normal, int32 occ_samples, Ray<T> &occ_rays);

  // I think I don't need a separate function to test for intersections of rays.
  // This should be part of triangle_intersection or other intersection methods when we get them.
  
  // Summing up the occlusion results can be a quick LAMBDA. 


  // ------------

  // These sampling methods can definitely be moved out of AmbientOcclusion.
  // These sampling methods were adapted from https://gitlab.kitware.com/mclarsen/vtk-m/blob/pathtracer/vtkm/rendering/raytracing/Sampler.h

  template <int32 Base>
  //static void Halton2D(const int32 &sampleNum, Vec<T,2> &coord);
  DRAY_EXEC static void Halton2D(const int32 &sampleNum, Vec<T,2> &coord);

  //static Vec<T,3> CosineWeightedHemisphere(const int32 &sampleNum);
  //static void ConstructTangentBasis( const Vec<T,3> &normal, Vec<T,3> &xAxis, Vec<T,3> &yAxis);
  DRAY_EXEC static Vec<T,3> CosineWeightedHemisphere(const int32 &sampleNum);
  DRAY_EXEC static void ConstructTangentBasis( const Vec<T,3> &normal, Vec<T,3> &xAxis, Vec<T,3> &yAxis);

};
} //namespace dray
#endif // DRAY_AMBIENT_OCCLUSION_HPP
