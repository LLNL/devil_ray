#include <dray/GridFunction/mesh.hpp>
#include <dray/aabb.hpp>
#include <dray/point_location.hpp>
#include <dray/policies.hpp>

#include <RAJA/RAJA.hpp>

namespace dray
{

namespace detail
{

template<typename T>
BVH construct_bvh(Mesh<T> &mesh)
{
  constexpr double bbox_scale = 1.000001;

  const int num_els = mesh.get_num_elem();

  constexpr int splits = 3;

  Array<AABB<>> aabbs;
  Array<int32> prim_ids;

  aabbs.resize(num_els*(splits+1));
  prim_ids.resize(num_els*(splits+1));

  AABB<> *aabb_ptr = aabbs.get_device_ptr();
  int32  *prim_ids_ptr = prim_ids.get_device_ptr();

  MeshAccess<T> device_mesh = mesh.access_device_mesh();

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, num_els), [=] DRAY_LAMBDA (int32 el_id)
  {
    AABB<> boxs[splits + 1];
    AABB<> ref_boxs[splits + 1];
    AABB<> tot;


    device_mesh.get_elem(el_id).get_bounds(boxs[0].m_ranges);
    tot = boxs[0];
    ref_boxs[0] = AABB<>::ref_universe();
    int32 count = 1;

    for(int i = 0; i < splits; ++i)
    {
      //find split
      int32 max_id = 0;
      float32 max_length = boxs[0].max_length();
      for(int b = 1; b < count; ++b)
      {
        float32 length = boxs[b].max_length();
        if(length > max_length)
        {
          max_id = b;
          max_length = length;
        }
      }

      int32 max_dim = boxs[max_id].max_dim();
      // split the reference box into two peices along largest phys dim
      ref_boxs[count] = ref_boxs[max_id].split(max_dim);

      // udpate the phys bounds
      device_mesh.get_elem(el_id).get_sub_bounds(ref_boxs[max_id].m_ranges,
                                                 boxs[max_id].m_ranges);
      device_mesh.get_elem(el_id).get_sub_bounds(ref_boxs[count].m_ranges,
                                                 boxs[count].m_ranges);
      count++;
    }

    AABB<> res;
    for(int i = 0; i < splits + 1; ++i)
    {
      boxs[i].scale(bbox_scale);
      res.include(boxs[i]);
      aabb_ptr[el_id * (splits + 1) + i] = boxs[i];
      prim_ids_ptr[el_id * (splits + 1) + i] = el_id;
    }

    if(el_id > 100 && el_id < 200)
    {
      printf("cell id %d AREA %f %f diff %f\n",
                                     el_id,
                                     tot.area(),
                                     res.area(),
                                     tot.area() - res.area());
      //AABB<> ol =  tot.intersect(res);
      //float32 overlap =  ol.area();

      //printf("overlap %f\n", overlap);
      //printf("%f %f %f - %f %f %f\n",
      //      tot.m_ranges[0].min(),
      //      tot.m_ranges[1].min(),
      //      tot.m_ranges[2].min(),
      //      tot.m_ranges[0].max(),
      //      tot.m_ranges[1].max(),
      //      tot.m_ranges[2].max());
    }

  });

  LinearBVHBuilder builder;
  BVH bvh = builder.construct(aabbs, prim_ids);
  std::cout<<"****** "<<bvh.m_bounds<<" "<<bvh.m_bounds.area()<<"\n";
  return bvh;
}

}

template <typename T, int32 dim>
const BVH Mesh<T,dim>::get_bvh() const
{
  return m_bvh;
}

template<typename T, int32 dim>
Mesh<T,dim>::Mesh(const GridFunctionData<T,dim> &dof_data, int32 poly_order)
  : m_dof_data(dof_data),
    m_poly_order(poly_order)
{
  m_bvh = detail::construct_bvh(*this);
}

template<typename T, int32 dim>
AABB<3>
Mesh<T,dim>::get_bounds() const
{
  return m_bvh.m_bounds;
}

template<typename T, int32 dim>
template <class StatsType>
void Mesh<T,dim>::locate(Array<int32> &active_idx,
                         Array<Vec<T,3>> &wpoints,
                         Array<RefPoint<T,3>> &rpoints,
                         StatsType &stats) const
{
  //template <int32 _RefDim>
  //using BShapeOp = BernsteinBasis<T,3>;
  //using ShapeOpType = BShapeOp<3>;

  const int32 size = wpoints.size();
  const int32 size_active = active_idx.size();
  // The results will go in rpoints. Make sure there's room.
  assert((rpoints.size() >= size_active));

  PointLocator locator(m_bvh);
  //constexpr int32 max_candidates = 5;
  constexpr int32 max_candidates = 100;
  //Size size_active * max_candidates.
  Array<int32> candidates = locator.locate_candidates(wpoints,
                                                      active_idx,
                                                      max_candidates);

  // For now the initial guess will always be the center of the element. TODO
  Vec<T,3> _ref_center;
  _ref_center = 0.5f;
  const Vec<T,3> ref_center = _ref_center;

  // Initialize outputs to well-defined dummy values.
  constexpr Vec<T,3> three_point_one_four = {3.14, 3.14, 3.14};

  // Assume that elt_ids and ref_pts are sized to same length as wpoints.
  //assert(elt_ids.size() == ref_pts.size());

  const int32    *active_idx_ptr = active_idx.get_device_ptr_const();

  RefPoint<T,3> *rpoints_ptr = rpoints.get_device_ptr();

  const Vec<T,3> *wpoints_ptr     = wpoints.get_device_ptr_const();
  const int32    *candidates_ptr = candidates.get_device_ptr_const();

#ifdef DRAY_STATS
  stats::AppStatsAccess device_appstats = stats.get_device_appstats();
#endif

  MeshAccess<T> device_mesh = this->access_device_mesh();

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, size_active), [=] DRAY_LAMBDA (int32 aii)
  {
    const int32 ii = active_idx_ptr[aii];
    RefPoint<T,3> rpt = rpoints_ptr[ii];
    const Vec<T,3> target_pt = wpoints_ptr[ii];

    rpt.m_el_coords = three_point_one_four;
    rpt.m_el_id = -1;

    // - Use aii to index into candidates.
    // - Use ii to index into wpoints, elt_ids, and ref_pts.

    int32 count = 0;
    int32 el_idx = candidates_ptr[aii*max_candidates + count];
    Vec<T,3> el_coords = ref_center;

    // For accounting/debugging.
    AABB<> cand_overlap = AABB<>::universe();

    bool found_inside = false;
    int32 steps_taken = 0;
    while(!found_inside && count < max_candidates && el_idx != -1)
    {
      steps_taken = 0;
      const bool use_init_guess = false;

      // For accounting/debugging.
      AABB<> bbox;
      device_mesh.get_elem(el_idx).get_bounds(bbox.m_ranges);
      cand_overlap.intersect(bbox);

#ifdef DRAY_STATS
      stats::IterativeProfile iter_prof;
      iter_prof.construct();

      found_inside = device_mesh.world2ref(iter_prof,
                                           el_idx,
                                           target_pt,
                                           el_coords,
                                           use_init_guess);  // Much easier than before.
      steps_taken = iter_prof.m_num_iter;

      RAJA::atomic::atomicAdd<atomic_policy>(
          &device_appstats.m_query_stats_ptr[ii].m_total_tests, 1);

      RAJA::atomic::atomicAdd<atomic_policy>(
          &device_appstats.m_query_stats_ptr[ii].m_total_test_iterations,
          steps_taken);

      RAJA::atomic::atomicAdd<atomic_policy>(
          &device_appstats.m_elem_stats_ptr[el_idx].m_total_tests, 1);

      RAJA::atomic::atomicAdd<atomic_policy>(
          &device_appstats.m_elem_stats_ptr[el_idx].m_total_test_iterations,
          steps_taken);
#else
      found_inside = device_mesh.world2ref(el_idx,
                                           target_pt,
                                           el_coords,
                                           use_init_guess);
#endif

      if (!found_inside && count < max_candidates-1)
      {
        // Continue searching with the next candidate.
        count++;
        el_idx = candidates_ptr[aii*max_candidates + count];
      }
    }

    // After testing each candidate, now record the result.
    if (found_inside)
    {
      rpt.m_el_id = el_idx;
      rpt.m_el_coords = el_coords;
    }
    else
    {
      rpt.m_el_id = -1;
    }
    rpoints_ptr[ii] = rpt;

#ifdef DRAY_STATS
    if (found_inside)
    {
      RAJA::atomic::atomicAdd<atomic_policy>(
          &device_appstats.m_query_stats_ptr[ii].m_total_hits,
          1);

      RAJA::atomic::atomicAdd<atomic_policy>(
          &device_appstats.m_query_stats_ptr[ii].m_total_hit_iterations,
          steps_taken);

      RAJA::atomic::atomicAdd<atomic_policy>(
          &device_appstats.m_elem_stats_ptr[el_idx].m_total_hits,
          1);

      RAJA::atomic::atomicAdd<atomic_policy>(
          &device_appstats.m_elem_stats_ptr[el_idx].m_total_hit_iterations,
          steps_taken);
    }
#endif
  });
}

// Explicit instantiations.
template class MeshAccess<float32, 3>;
template class MeshAccess<float64, 3>;

// Explicit instantiations.
template class Mesh<float32, 3>;
template class Mesh<float64, 3>;
}
