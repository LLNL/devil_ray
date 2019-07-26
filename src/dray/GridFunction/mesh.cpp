#include <dray/GridFunction/mesh.hpp>
#include <dray/array_utils.hpp>
#include <dray/aabb.hpp>
#include <dray/point_location.hpp>
#include <dray/policies.hpp>

#include <dray/Element/pos_tensor_element.hpp>
#include <dray/Element/pos_simplex_element.hpp>

#include <RAJA/RAJA.hpp>

namespace dray
{

namespace detail
{

DRAY_EXEC void swap(int32 &a, int32 &b)
{
  int32 tmp;
  tmp = a;
  a = b;
  b = tmp;
}

DRAY_EXEC void sort4(Vec<int32,4> &vec)
{
  if(vec[0] > vec[1])
  {
    swap(vec[0], vec[1]);
  }
  if(vec[2] > vec[3])
  {
    swap(vec[2], vec[3]);
  }
  if(vec[0] > vec[2])
  {
    swap(vec[0], vec[2]);
  }
  if(vec[1] > vec[3])
  {
    swap(vec[1], vec[3]);
  }
  if(vec[1] > vec[2])
  {
    swap(vec[1], vec[2]);
  }
}

template<typename T>
void reorder(Array<int32> &indices, Array<T> &array)
{
  assert(indices.size() == array.size());
  const int size = array.size();

  Array<T> temp;
  temp.resize(size);

  T *temp_ptr = temp.get_device_ptr();
  const T *array_ptr = array.get_device_ptr_const();
  const int32 *indices_ptr = indices.get_device_ptr_const();

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, size), [=] DRAY_LAMBDA (int32 i)
  {
    int32 in_idx = indices_ptr[i];
    temp_ptr[i] = array_ptr[in_idx];
  });

  array = temp;
}

Array<int32> sort_faces(Array<Vec<int32,4>> &faces)
{
  const int size = faces.size();
  Array<int32> iter = array_counting(size, 0, 1);
  // TODO: create custom sort for GPU / CPU
  int32  *iter_ptr = iter.get_host_ptr();
  Vec<int32,4> *faces_ptr = faces.get_host_ptr();

  std::sort(iter_ptr,
            iter_ptr + size,
            [=](int32 i1, int32 i2)
            {
              const Vec<int32,4> f1 = faces_ptr[i1];
              const Vec<int32,4> f2 = faces_ptr[i2];
              if(f1[0] == f2[0])
              {
                if(f1[1] == f2[1])
                {
                  if(f1[2] == f2[2])
                  {
                    return f1[3] < f2[3];
                  }
                  else
                  {
                   return f1[2] < f2[2];
                  }
                }
                else
                {
                 return f1[1] < f2[1];
                }
              }
              else
              {
               return f1[0] < f2[0];
              }

              //return true;
            });


  reorder(iter, faces);
  return iter;
}

DRAY_EXEC bool is_same(const Vec<int32,4> &a, const Vec<int32,4> &b)
{
  return (a[0] == b[0]) &&
         (a[1] == b[1]) &&
         (a[2] == b[2]) &&
         (a[3] == b[3]);
}

void unique_faces(Array<Vec<int32,4>> &faces, Array<int32> &orig_ids)
{
  const int32 size = faces.size();

  Array<int32> unique_flags;
  unique_flags.resize(size);
  // assum everyone is unique
  array_memset(unique_flags, 1);

  const Vec<int32,4> *faces_ptr = faces.get_device_ptr_const();
  int32 *unique_flags_ptr = unique_flags.get_device_ptr();

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, size), [=] DRAY_LAMBDA (int32 i)
  {
    // we assume everthing is sorted and there can be at most
    // two faces that can be shared
    const Vec<int32,4> me = faces_ptr[i];
    bool duplicate = false;
    if(i != 0)
    {
      const Vec<int32,4> left = faces_ptr[i-1];
      if(is_same(me,left))
      {
        duplicate = true;
      }
    }
    if(i != size - 1)
    {
      const Vec<int32,4> right = faces_ptr[i+1];
      if(is_same(me,right))
      {
        duplicate = true;
      }
    }

    if(duplicate)
    {
      // mark myself for death
      unique_flags_ptr[i] = 0;
    }
  });
  faces = index_flags(unique_flags, faces);
  orig_ids = index_flags(unique_flags, orig_ids);
}

template<typename T, class ElemT>
Array<Vec<int32,4>> extract_faces(Mesh<T, ElemT> &mesh)
{
  const int num_els = mesh.get_num_elem();

  Array<Vec<int32,4>> faces;
  faces.resize(num_els * 6);
  Vec<int32,4> *faces_ptr = faces.get_device_ptr();

  MeshAccess<T, ElemT> device_mesh = mesh.access_device_mesh();
  RAJA::forall<for_policy>(RAJA::RangeSegment(0, num_els), [=] DRAY_LAMBDA (int32 el_id)
  {
    // assume that if one dof is shared on a face then all dofs are shares.
    // if this is not the case this is a much harder problem
    const int32 p = device_mesh.m_poly_order;
    const int32 stride_y = p + 1;
    const int32 stride_z = stride_y * stride_y;
    const int32 el_offset = stride_z * stride_y * el_id;
    const int32 *el_ptr = device_mesh.m_idx_ptr + el_offset;
    int32 corners[8];
    corners[0] = el_ptr[0];
    corners[1] = el_ptr[p];
    corners[2] = el_ptr[stride_y * p];
    corners[3] = el_ptr[stride_y * p + p];
    corners[4] = el_ptr[stride_z * p];
    corners[5] = el_ptr[stride_z * p + p];
    corners[6] = el_ptr[stride_z * p + stride_y * p];
    corners[7] = el_ptr[stride_z * p + stride_y * p + p];

    // I think this is following masado's conventions
    Vec<int32,4> face;

    // x
    face[0] = corners[0];
    face[1] = corners[2];
    face[2] = corners[4];
    face[3] = corners[6];
    sort4(face);

    faces_ptr[el_id * 6 + 0] = face;
    // X
    face[0] = corners[1];
    face[1] = corners[3];
    face[2] = corners[5];
    face[3] = corners[7];

    sort4(face);
    faces_ptr[el_id * 6 + 3] = face;

    // y
    face[0] = corners[0];
    face[1] = corners[1];
    face[2] = corners[4];
    face[3] = corners[5];

    sort4(face);
    faces_ptr[el_id * 6 + 1] = face;
    // Y
    face[0] = corners[2];
    face[1] = corners[3];
    face[2] = corners[6];
    face[3] = corners[7];

    sort4(face);
    faces_ptr[el_id * 6 + 4] = face;

    // z
    face[0] = corners[0];
    face[1] = corners[1];
    face[2] = corners[2];
    face[3] = corners[3];

    sort4(face);
    faces_ptr[el_id * 6 + 2] = face;

    // Z
    face[0] = corners[4];
    face[1] = corners[5];
    face[2] = corners[6];
    face[3] = corners[7];

    sort4(face);
    faces_ptr[el_id * 6 + 5] = face;
  });

  return faces;
}

Array<Vec<int32,2>> reconstruct(const int num_elements, Array<int32> &orig_ids)
{
  const int32 size = orig_ids.size();

  Array<Vec<int32,2>> face_ids;
  face_ids.resize(size);

  const int32 *orig_ids_ptr = orig_ids.get_device_ptr_const();
  Vec<int32,2> *faces_ids_ptr = face_ids.get_device_ptr();

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, size), [=] DRAY_LAMBDA (int32 i)
  {
    const int32 flat_id = orig_ids_ptr[i];
    const int32 el_id = flat_id / 6;
    const int32 face_id = flat_id % 6;
    Vec<int32,2> face;
    face[0] = el_id;
    face[1] = face_id;
    faces_ids_ptr[i] = face;
  });
  return face_ids;
}

//TODO
/// template<typename T, class ElemT>
/// BVH construct_face_bvh(Mesh<T, ElemT> &mesh, Array<Vec<int32,2>> &faces)
/// {
///   constexpr double bbox_scale = 1.000001;
///   const int32 num_faces = faces.size();
///   Array<AABB<>> aabbs;
///   aabbs.resize(num_faces);
///   AABB<> *aabb_ptr = aabbs.get_device_ptr();
/// 
///   MeshAccess<T, ElemT> device_mesh = mesh.access_device_mesh();
///   const Vec<int32,2> *faces_ptr = faces.get_device_ptr_const();
/// 
///   RAJA::forall<for_policy>(RAJA::RangeSegment(0, num_faces), [=] DRAY_LAMBDA (int32 face_id)
///   {
///     const Vec<int32,2> face = faces_ptr[face_id];
///     FaceElement<T,3> face_elem = device_mesh.get_elem(face[0]).get_face_element(face[1]);
/// 
///     AABB<> bounds;
///     face_elem.get_bounds(bounds);
///     bounds.scale(bbox_scale);
///     aabb_ptr[face_id] = bounds;
///   });
/// 
///   LinearBVHBuilder builder;
///   BVH bvh = builder.construct(aabbs);
///   return bvh;
/// }

//TODO
/// template<typename T, class ElemT>
/// typename Mesh<T, ElemT>::ExternalFaces  external_faces(Mesh<T, ElemT> &mesh)
/// {
///   Array<Vec<int32,4>> faces = extract_faces(mesh);
/// 
///   Array<int32> orig_ids = sort_faces(faces);
///   unique_faces(faces, orig_ids);
/// 
/// 
///   const int num_els = mesh.get_num_elem();
///   Array<Vec<int32,2>> res = reconstruct(num_els, orig_ids);
/// 
///   BVH bvh = construct_face_bvh(mesh, res);
/// 
///   typename Mesh<T, ElemT>::ExternalFaces ext_faces;
///   ext_faces.m_faces = res;
///   ext_faces.m_bvh = bvh;
///   return ext_faces;
/// }

template<typename T, class ElemT>
BVH construct_bvh(Mesh<T, ElemT> &mesh, Array<AABB<ElemT::get_dim()>> &ref_aabbs)
{
  constexpr uint32 dim = ElemT::get_dim();

  constexpr double bbox_scale = 1.000001;

  const int num_els = mesh.get_num_elem();

  constexpr int splits = 1;

  Array<AABB<>> aabbs;
  Array<int32> prim_ids;

  aabbs.resize(num_els*(splits+1));
  prim_ids.resize(num_els*(splits+1));
  ref_aabbs.resize(num_els*(splits+1));

  AABB<> *aabb_ptr = aabbs.get_device_ptr();
  int32  *prim_ids_ptr = prim_ids.get_device_ptr();
  AABB<dim> *ref_aabbs_ptr = ref_aabbs.get_device_ptr();

  MeshAccess<T, ElemT> device_mesh = mesh.access_device_mesh();

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, num_els), [=] DRAY_LAMBDA (int32 el_id)
  {
    AABB<> boxs[splits + 1];
    AABB<dim> ref_boxs[splits + 1];
    AABB<> tot;

    device_mesh.get_elem(el_id).get_bounds(boxs[0]);
    tot = boxs[0];
    ref_boxs[0] = AABB<dim>::ref_universe();
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
      device_mesh.get_elem(el_id).get_sub_bounds(ref_boxs[max_id],
                                                 boxs[max_id]);
      device_mesh.get_elem(el_id).get_sub_bounds(ref_boxs[count],
                                                 boxs[count]);
      count++;
    }

    AABB<> res;
    for(int i = 0; i < splits + 1; ++i)
    {
      boxs[i].scale(bbox_scale);
      res.include(boxs[i]);
      aabb_ptr[el_id * (splits + 1) + i] = boxs[i];
      prim_ids_ptr[el_id * (splits + 1) + i] = el_id;
      ref_aabbs_ptr[el_id * (splits + 1) + i] = ref_boxs[i];
    }

    //if(el_id > 100 && el_id < 200)
    //{
    //  printf("cell id %d AREA %f %f diff %f\n",
    //                                 el_id,
    //                                 tot.area(),
    //                                 res.area(),
    //                                 tot.area() - res.area());
    //  //AABB<> ol =  tot.intersect(res);
    //  //float32 overlap =  ol.area();

    //  //printf("overlap %f\n", overlap);
    //  //printf("%f %f %f - %f %f %f\n",
    //  //      tot.m_ranges[0].min(),
    //  //      tot.m_ranges[1].min(),
    //  //      tot.m_ranges[2].min(),
    //  //      tot.m_ranges[0].max(),
    //  //      tot.m_ranges[1].max(),
    //  //      tot.m_ranges[2].max());
    //}

  });

  LinearBVHBuilder builder;
  BVH bvh = builder.construct(aabbs, prim_ids);
  return bvh;
}

}

template <typename T, class ElemT>
const BVH Mesh<T, ElemT>::get_bvh() const
{
  return m_bvh;
}

template<typename T, class ElemT>
Mesh<T, ElemT>::Mesh(const GridFunctionData<T,3u> &dof_data, int32 poly_order)
  : m_dof_data(dof_data),
    m_poly_order(poly_order)
{
  m_bvh = detail::construct_bvh(*this, m_ref_aabbs);
  /// m_external_faces = detail::external_faces(*this);  // TODO face_mesh
}

template<typename T, class ElemT>
AABB<3>
Mesh<T, ElemT>::get_bounds() const
{
  return m_bvh.m_bounds;
}

template<typename T, class ElemT>
template <class StatsType>
void Mesh<T, ElemT>::locate(Array<int32> &active_idx,
                         Array<Vec<T,3u>> &wpoints,
                         Array<RefPoint<T,dim>> &rpoints,
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
  PointLocator::Candidates candidates = locator.locate_candidates(wpoints,
                                                                  active_idx,
                                                                  max_candidates);

  const AABB<dim> *ref_aabb_ptr = m_ref_aabbs.get_device_ptr_const();

  // Initialize outputs to well-defined dummy values.
  Vec<T,dim> three_point_one_four;   three_point_one_four = 3.14;

  // Assume that elt_ids and ref_pts are sized to same length as wpoints.
  //assert(elt_ids.size() == ref_pts.size());

  const int32    *active_idx_ptr = active_idx.get_device_ptr_const();

  RefPoint<T,dim> *rpoints_ptr = rpoints.get_device_ptr();

  const Vec<T,3> *wpoints_ptr = wpoints.get_device_ptr_const();
  const int32    *cell_id_ptr = candidates.m_candidates.get_device_ptr_const();
  const int32    *aabb_id_ptr = candidates.m_aabb_ids.get_device_ptr_const();

#ifdef DRAY_STATS
  stats::AppStatsAccess device_appstats = stats.get_device_appstats();

  Array<stats::MattStats> mstats;
  mstats.resize(size);
  stats::MattStats *mstats_ptr = mstats.get_device_ptr();
#endif

  MeshAccess<T, ElemT> device_mesh = this->access_device_mesh();

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, size_active), [=] DRAY_LAMBDA (int32 aii)
  {
#ifdef DRAY_STATS
    stats::MattStats mstat;
    mstat.construct();
#endif
    const int32 ii = active_idx_ptr[aii];
    RefPoint<T,dim> rpt = rpoints_ptr[ii];
    const Vec<T,3> target_pt = wpoints_ptr[ii];

    rpt.m_el_coords = three_point_one_four;
    rpt.m_el_id = -1;

    // - Use aii to index into candidates.
    // - Use ii to index into wpoints, elt_ids, and ref_pts.

    int32 count = 0;
    int32 el_idx = cell_id_ptr[aii*max_candidates + count];
    int32 aabb_idx = aabb_id_ptr[aii*max_candidates + count];
    AABB<dim> ref_start_box = ref_aabb_ptr[aabb_idx];
    Vec<T,dim> el_coords;
    // For accounting/debugging.
    AABB<> cand_overlap = AABB<>::universe();

    bool found_inside = false;
    int32 steps_taken = 0;
    while(!found_inside && count < max_candidates && el_idx != -1)
    {
      steps_taken = 0;
      const bool use_init_guess = true;

      // For accounting/debugging.
      AABB<> bbox;
      device_mesh.get_elem(el_idx).get_bounds(bbox);
      cand_overlap.intersect(bbox);

#ifdef DRAY_STATS
      stats::IterativeProfile iter_prof;
      iter_prof.construct();
      mstat.m_candidates++;

      found_inside = device_mesh.get_elem(el_idx).eval_inverse(iter_prof,
                                           target_pt,
                                           ref_start_box,
                                           el_coords,
                                           use_init_guess);  // Much easier than before.
      steps_taken = iter_prof.m_num_iter;
      mstat.m_newton_iters += steps_taken;

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
      found_inside = device_mesh.get_elem(el_idx).eval_inverse(
                                           target_pt,
                                           ref_start_box,
                                           el_coords,
                                           use_init_guess);
#endif

      if (!found_inside && count < max_candidates-1)
      {
        // Continue searching with the next candidate.
        count++;
        el_idx = cell_id_ptr[aii*max_candidates + count];
        aabb_idx = aabb_id_ptr[aii*max_candidates + count];
        ref_start_box = ref_aabb_ptr[aabb_idx];
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
      mstat.m_found = 1;

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
    mstats_ptr[aii] = mstat;
#endif
  });

#ifdef DRAY_STATS
  stats::StatStore::add_point_stats(wpoints, mstats);
#endif
}


// Explicit instantiations.
template class MeshAccess<float32, MeshElem<float32, 2u, ElemType::Quad, Order::General>>;
template class MeshAccess<float64, MeshElem<float64, 2u, ElemType::Quad, Order::General>>;
template class MeshAccess<float32, MeshElem<float32, 2u, ElemType::Tri, Order::General>>;
template class MeshAccess<float64, MeshElem<float64, 2u, ElemType::Tri, Order::General>>;

template class MeshAccess<float32, MeshElem<float32, 3u, ElemType::Quad, Order::General>>;
template class MeshAccess<float32, MeshElem<float32, 3u, ElemType::Tri, Order::General>>;
template class MeshAccess<float64, MeshElem<float64, 3u, ElemType::Quad, Order::General>>;
template class MeshAccess<float64, MeshElem<float64, 3u, ElemType::Tri, Order::General>>;

// Explicit instantiations.
/// template class Mesh<float32, MeshElem<float32, 2u, ElemType::Quad, Order::General>>;  //TODO bar locate() from 2x3
/// template class Mesh<float64, MeshElem<float64, 2u, ElemType::Quad, Order::General>>;
/// template class Mesh<float32, MeshElem<float32, 2u, ElemType::Tri, Order::General>>;
/// template class Mesh<float64, MeshElem<float64, 2u, ElemType::Tri, Order::General>>;

template class Mesh<float32, MeshElem<float32, 3u, ElemType::Quad, Order::General>>;
template class Mesh<float64, MeshElem<float64, 3u, ElemType::Quad, Order::General>>;
/// template class Mesh<float32, MeshElem<float32, 3u, ElemType::Tri, Order::General>>;   //TODO change ref boxes to SubRef<etype>
/// template class Mesh<float64, MeshElem<float64, 3u, ElemType::Tri, Order::General>>;
}
