#ifndef DRAY_AMBIENT_OCCLUSION_HPP
#define DRAY_AMBIENT_OCCLUSION_HPP

#include <dray/intersection_context.hpp>

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

  static T nudge_dist;

  /**
   * [in] hit_points
   * [in] surface_normals
   * [in] occ_samples
   * [out] occ_component
   */
  // TODO Ask if types should be adjusted.
  //template<typename C>  // for color factor.
  static
  //void calc_occlusion(Array<Vec<T,3>> &hit_points, Array<Vec<T,3>> &surface_normals, int32 occ_samples, Array<C> &occ_component);
  //void calc_occlusion(Ray<T> &incoming_rays, int32 occ_samples, Array<C> &occ_component);
  Ray<T> gen_occlusion(IntersectionContext<T> intersection_ctx, int32 occ_samples, T occ_dist);

  // Note: We return type Ray<T> instead of [out] parameter, because the calling code
  // does not know how many occlusion rays there will be. (It will be a multiple of
  // the number of valid primary intersections, but the calling code does not know how
  // many valid primary intersections there are.)

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
