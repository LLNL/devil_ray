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

  const static T nudge_dist;

  /**
   * [in] intersection_ctx
   * [in] occ_samples
   * [in] occ_near
   * [in] occ_far
   * 
   * returns occ_rays
   */
  static
  Ray<T> gen_occlusion(const IntersectionContext<T> intersection_ctx, const int32 occ_samples, const T occ_near, const T occ_far);
  static
  Ray<T> gen_occlusion(const IntersectionContext<T> intersection_ctx, const int32 occ_samples, const T occ_near, const T occ_far, Array<int32> &compact_indexing);
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
