// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <dray/Element/element.hpp>
#include <dray/GridFunction/device_mesh.hpp>
#include <dray/GridFunction/mesh.hpp>
#include <dray/GridFunction/mesh_utils.hpp>
#include <dray/utils/data_logger.hpp>

#include <dray/aabb.hpp>
#include <dray/array_utils.hpp>
#include <dray/dray.hpp>

#include <RAJA/RAJA.hpp>
#include <dray/policies.hpp>
#include <dray/error_check.hpp>


namespace dray
{

namespace detail
{

DRAY_EXEC void swap (int32 &a, int32 &b)
{
  int32 tmp;
  tmp = a;
  a = b;
  b = tmp;
}

DRAY_EXEC void sort4 (Vec<int32, 4> &vec)
{
  if (vec[0] > vec[1])
  {
    swap (vec[0], vec[1]);
  }
  if (vec[2] > vec[3])
  {
    swap (vec[2], vec[3]);
  }
  if (vec[0] > vec[2])
  {
    swap (vec[0], vec[2]);
  }
  if (vec[1] > vec[3])
  {
    swap (vec[1], vec[3]);
  }
  if (vec[1] > vec[2])
  {
    swap (vec[1], vec[2]);
  }
}

template <typename T> void reorder (Array<int32> &indices, Array<T> &array)
{
  assert (indices.size () == array.size ());
  const int size = array.size ();

  Array<T> temp;
  temp.resize (size);

  T *temp_ptr = temp.get_device_ptr ();
  const T *array_ptr = array.get_device_ptr_const ();
  const int32 *indices_ptr = indices.get_device_ptr_const ();

  RAJA::forall<for_policy> (RAJA::RangeSegment (0, size), [=] DRAY_LAMBDA (int32 i) {
    int32 in_idx = indices_ptr[i];
    temp_ptr[i] = array_ptr[in_idx];
  });
  DRAY_ERROR_CHECK();

  array = temp;
}

Array<int32> sort_faces (Array<Vec<int32, 4>> &faces)
{
  const int size = faces.size ();
  Array<int32> iter = array_counting (size, 0, 1);
  // TODO: create custom sort for GPU / CPU
  int32 *iter_ptr = iter.get_host_ptr ();
  Vec<int32, 4> *faces_ptr = faces.get_host_ptr ();

  std::sort (iter_ptr, iter_ptr + size, [=] (int32 i1, int32 i2) {
    const Vec<int32, 4> f1 = faces_ptr[i1];
    const Vec<int32, 4> f2 = faces_ptr[i2];
    if (f1[0] == f2[0])
    {
      if (f1[1] == f2[1])
      {
        if (f1[2] == f2[2])
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

    // return true;
  });


  reorder (iter, faces);
  return iter;
}

DRAY_EXEC bool is_same (const Vec<int32, 4> &a, const Vec<int32, 4> &b)
{
  return (a[0] == b[0]) && (a[1] == b[1]) && (a[2] == b[2]) && (a[3] == b[3]);
}

void unique_faces (Array<Vec<int32, 4>> &faces, Array<int32> &orig_ids)
{
  const int32 size = faces.size ();

  Array<int32> unique_flags;
  unique_flags.resize (size);
  // assum everyone is unique
  array_memset (unique_flags, 1);

  const Vec<int32, 4> *faces_ptr = faces.get_device_ptr_const ();
  int32 *unique_flags_ptr = unique_flags.get_device_ptr ();

  RAJA::forall<for_policy> (RAJA::RangeSegment (0, size), [=] DRAY_LAMBDA (int32 i) {
    // we assume everthing is sorted and there can be at most
    // two faces that can be shared
    const Vec<int32, 4> me = faces_ptr[i];
    bool duplicate = false;
    if (i != 0)
    {
      const Vec<int32, 4> left = faces_ptr[i - 1];
      if (is_same (me, left))
      {
        duplicate = true;
      }
    }
    if (i != size - 1)
    {
      const Vec<int32, 4> right = faces_ptr[i + 1];
      if (is_same (me, right))
      {
        duplicate = true;
      }
    }

    if (duplicate)
    {
      // mark myself for death
      unique_flags_ptr[i] = 0;
    }
  });
  DRAY_ERROR_CHECK();
  faces = index_flags (unique_flags, faces);
  orig_ids = index_flags (unique_flags, orig_ids);
}

// TODO extract_faces() needs to be extended to triangular/tetrahedral meshes too.
template <class ElemT> Array<Vec<int32, 4>> extract_faces (Mesh<ElemT> &mesh)
{
  const int num_els = mesh.get_num_elem ();

  Array<Vec<int32, 4>> faces;
  faces.resize (num_els * 6);
  Vec<int32, 4> *faces_ptr = faces.get_device_ptr ();

  DeviceMesh<ElemT> device_mesh (mesh);

  RAJA::forall<for_policy> (RAJA::RangeSegment (0, num_els), [=] DRAY_LAMBDA (int32 el_id) {
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
    Vec<int32, 4> face;

    // x
    face[0] = corners[0];
    face[1] = corners[2];
    face[2] = corners[4];
    face[3] = corners[6];
    sort4 (face);

    faces_ptr[el_id * 6 + 0] = face;
    // X
    face[0] = corners[1];
    face[1] = corners[3];
    face[2] = corners[5];
    face[3] = corners[7];

    sort4 (face);
    faces_ptr[el_id * 6 + 3] = face;

    // y
    face[0] = corners[0];
    face[1] = corners[1];
    face[2] = corners[4];
    face[3] = corners[5];

    sort4 (face);
    faces_ptr[el_id * 6 + 1] = face;
    // Y
    face[0] = corners[2];
    face[1] = corners[3];
    face[2] = corners[6];
    face[3] = corners[7];

    sort4 (face);
    faces_ptr[el_id * 6 + 4] = face;

    // z
    face[0] = corners[0];
    face[1] = corners[1];
    face[2] = corners[2];
    face[3] = corners[3];

    sort4 (face);
    faces_ptr[el_id * 6 + 2] = face;

    // Z
    face[0] = corners[4];
    face[1] = corners[5];
    face[2] = corners[6];
    face[3] = corners[7];

    sort4 (face);
    faces_ptr[el_id * 6 + 5] = face;
  });
  DRAY_ERROR_CHECK();

  return faces;
}

// Returns faces, where faces[i][0] = el_id and 0 <= faces[i][1] = face_id < 6.
Array<Vec<int32, 2>> reconstruct (Array<int32> &orig_ids)
{
  const int32 size = orig_ids.size ();

  Array<Vec<int32, 2>> face_ids;
  face_ids.resize (size);

  const int32 *orig_ids_ptr = orig_ids.get_device_ptr_const ();
  Vec<int32, 2> *faces_ids_ptr = face_ids.get_device_ptr ();

  RAJA::forall<for_policy> (RAJA::RangeSegment (0, size), [=] DRAY_LAMBDA (int32 i) {
    const int32 flat_id = orig_ids_ptr[i];
    const int32 el_id = flat_id / 6;
    const int32 face_id = flat_id % 6;
    Vec<int32, 2> face;
    face[0] = el_id;
    face[1] = face_id;
    faces_ids_ptr[i] = face;
  });
  DRAY_ERROR_CHECK();
  return face_ids;
}

// TODO
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

// TODO
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
///   Array<Vec<int32,2>> res = reconstruct(orig_ids);
///
///   BVH bvh = construct_face_bvh(mesh, res);
///
///   typename Mesh<T, ElemT>::ExternalFaces ext_faces;
///   ext_faces.m_faces = res;
///   ext_faces.m_bvh = bvh;
///   return ext_faces;
/// }

template <class ElemT>
BVH construct_bvh (Mesh<ElemT> &mesh, Array<AABB<ElemT::get_dim ()>> &ref_aabbs)
{
  DRAY_LOG_OPEN ("construct_bvh");
  constexpr uint32 dim = ElemT::get_dim ();

  constexpr double bbox_scale = 1.000001;

  const int num_els = mesh.get_num_elem ();

  // constexpr int splits = 2 * (2 << dim);
  constexpr int splits = 62;
#warning "splits no longer controlable"

  Array<AABB<>> aabbs;
  Array<int32> prim_ids;

  aabbs.resize (num_els * (splits + 1));
  prim_ids.resize (num_els * (splits + 1));
  ref_aabbs.resize (num_els * (splits + 1));

  printf("num boxes %d\n", aabbs.size());

  AABB<> *aabb_ptr = aabbs.get_device_ptr ();
  int32 *prim_ids_ptr = prim_ids.get_device_ptr ();
  AABB<dim> *ref_aabbs_ptr = ref_aabbs.get_device_ptr ();

  DeviceMesh<ElemT> device_mesh (mesh);

  Timer timer;
  RAJA::forall<for_policy> (RAJA::RangeSegment (0, num_els), [=] DRAY_LAMBDA (int32 el_id) {
    AABB<> boxs[splits + 1];
    AABB<dim> ref_boxs[splits + 1];
    AABB<> tot;

    device_mesh.get_elem (el_id).get_bounds (boxs[0]);
    tot = boxs[0];
    ref_boxs[0] = AABB<dim>::ref_universe ();
    int32 count = 1;

    for (int i = 0; i < splits; ++i)
    {
      // find split
      int32 max_id = 0;
      float32 max_length = boxs[0].max_length ();
      for (int b = 1; b < count; ++b)
      {
        float32 length = boxs[b].max_length ();
        if (length > max_length)
        {
          max_id = b;
          max_length = length;
        }
      }

      int32 max_dim = ref_boxs[max_id].max_dim ();
      // split the reference box into two peices along largest ref dim
      // Don't use the largest phys dim unless know how to match ref dim and phys dim.
      ref_boxs[count] = ref_boxs[max_id].split (max_dim);

      // udpate the phys bounds
      device_mesh.get_elem (el_id).get_sub_bounds (ref_boxs[max_id], boxs[max_id]);
      device_mesh.get_elem (el_id).get_sub_bounds (ref_boxs[count], boxs[count]);
      count++;
    }

    AABB<> res;
    for (int i = 0; i < splits + 1; ++i)
    {
      boxs[i].scale (bbox_scale);
      res.include (boxs[i]);
      aabb_ptr[el_id * (splits + 1) + i] = boxs[i];
      prim_ids_ptr[el_id * (splits + 1) + i] = el_id;
      ref_aabbs_ptr[el_id * (splits + 1) + i] = ref_boxs[i];
    }

    // if(el_id > 100 && el_id < 200)
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
  DRAY_LOG_ENTRY("aabb_extraction", timer.elapsed());
  DRAY_ERROR_CHECK();

  LinearBVHBuilder builder;
  BVH bvh = builder.construct (aabbs, prim_ids);
  DRAY_LOG_CLOSE ();
  return bvh;
}

template<typename ElemT>
DRAY_EXEC void wangs_formula(const DeviceMesh<ElemT> &device_mesh, float32 tolerance, int32 el_id, Vec<int32, ElemT::get_dim ()> &recursive_splits) {
  // Run a reduce loop to get the max number of splits
  // for each element:
      // run wang's formula on the edges
      // n = num of control points/dofs?
      // m = Max (for 0 to n-2) of |P_k - 2P_(k+1) + P_(k+2)|
      // splits = log_4(n(n-1)m / (8*tolerance))

  constexpr uint32 dim = ElemT::get_dim ();

  const int32 p = device_mesh.m_poly_order;
  const int32 stride_y = p + 1;
  const int32 stride_z = dim == 3 ? stride_y * stride_y : 0;
  const int32 el_offset = stride_z * stride_y * el_id;
  const int32 *el_ptr = device_mesh.m_idx_ptr + el_offset;
  const int32 strides [3] = {1, stride_y, stride_z};


  const int32 n = p + 1; 
  Vec<float32, dim> max_ms; 
  max_ms.zero();
  // m = Max (for 0 to n-2) of |P_k - 2P_(k+1) + P_(k+2)|

  // Iterate over the three dimensions
  for (int32 i = 0; i < n; ++i) { 
    for (int32 j = 0; j < n; ++j) {
      for (int32 d = 0; d < dim; d++) {
        // calculate m for each dimension m_i = |P_k - 2P_(k+1) + P_(k+2)|
        int32 outer_dim1 = (d + 1) % 3;
        int32 outer_dim2 = (d + 2) % 3;
        int32 start_idx = (strides[outer_dim1] * i) + (strides[outer_dim2] * j);
        for (int32 k = 0; k < n-2; ++k) {
          int32 step = strides[d];
          float32 current_m = (device_mesh.m_val_ptr[el_ptr[k * step + start_idx]] - 
            (device_mesh.m_val_ptr[el_ptr[(k+1) * step + start_idx]] * 2) + 
            device_mesh.m_val_ptr[el_ptr[(k+2) * step + start_idx]]).magnitude();
          if (max_ms[d] < current_m)
            max_ms[d] = current_m;
        }
      }
    }
  }

  for (int32 d = 0; d < dim; d++) {
    // recursive_splits = log_4(n(n-1)m / (8*tolerance))
    int32 rsplits = ceil(logf(n * (n - 1) * max_ms[d] / (8.0 * tolerance)) / logf(4));
    recursive_splits[d] = rsplits > 0 ? rsplits : 0;
  }




  //     for (int32 k = 0; k < n - 2; ++k) {
  //       step_x = (i * stride_y) + (j * stride_z) + k;
  //       m_x = (device_mesh.m_val_ptr[el_ptr[step_x]] - 
  //       (device_mesh.m_val_ptr[el_ptr[step_x + 1]] * 2) + 
  //       device_mesh.m_val_ptr[el_ptr[step_x + 2]]).magnitude();
  //       if (max_m_x < m_x) {
  //         max_m_x = m_x;
  //       }

  //       step_y = i + (j * stride_z) + (k * stride_y);
  //       m_y = (device_mesh.m_val_ptr[el_ptr[step_y]] - 
  //       (device_mesh.m_val_ptr[el_ptr[step_y + stride_y]] * 2) + 
  //       device_mesh.m_val_ptr[el_ptr[step_y + (2 * stride_y)]]).magnitude();
  //       if (max_m_y < m_y) {
  //         max_m_y = m_y;
  //       }

  //       if (dim == 3) {
  //         step_z = i + (j * stride_y) + (k * stride_z);
  //         m_z = (device_mesh.m_val_ptr[el_ptr[step_z]] - 
  //         (device_mesh.m_val_ptr[el_ptr[step_z + stride_z]] * 2) + 
  //         device_mesh.m_val_ptr[el_ptr[step_z + (2 * stride_z)]]).magnitude();
  //         if (max_m_z < m_z) {
  //           max_m_z = m_z;
  //         }
  //       }
  //     }
  //     if (dim == 2) {
  //       break;
  //     }
  //   }
  // }

  // recursive_splits = log_4(n(n-1)m / (8*tolerance))
  // int32 rsplits = ceil(logf(n * (n - 1) * max_m_x / (8.0 * tolerance)) / logf(4));
  // recursive_splits[0] = rsplits > 0 ? rsplits : 0;

  // rsplits = ceil(logf(n * (n - 1) * max_m_y / (8.0 * tolerance)) / logf(4));
  // recursive_splits[1] = rsplits > 0 ? rsplits : 0;

  // if (dim == 3) {
  //   rsplits = ceil(logf(n * (n - 1) * max_m_z / (8.0 * tolerance)) / logf(4));
  //   recursive_splits[2] = rsplits > 0 ? rsplits : 0;
  // }
}


template <class ElemT>
int32 get_wang_recursive_splits(Mesh<ElemT> &mesh, Array<int32> &el_num_boxes, Array<int32> &el_splits_dim, Array<int32> &offsets) {
  constexpr uint32 dim = ElemT::get_dim ();
  
  const int32 num_els = mesh.get_num_elem();
  const float32 flat_tol = dray::get_zone_flatness_tolerance();

  el_splits_dim.resize(dim * num_els); // For each dimention
  el_num_boxes.resize(num_els);
  offsets.resize(num_els);

  int32 *el_splits_dim_ptr = el_splits_dim.get_device_ptr();
  int32 *el_num_boxes_ptr = el_num_boxes.get_device_ptr();
  DeviceMesh<ElemT> device_mesh (mesh);

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, num_els), [=] DRAY_LAMBDA (int32 el_id) 
  {
    // Get the number of recursive splits needed to be made in each dimension.
    Vec<int32, dim> dim_splits;
    int32 total_recursive_splits = 0;
    wangs_formula(device_mesh, flat_tol, el_id, dim_splits);
    for (int d = 0; d < dim; ++d) {
      el_splits_dim_ptr[(dim * el_id) + d] = dim_splits[d];  
      total_recursive_splits += dim_splits[d];
    }

    // 2^splits to get the total number of boxes for a single dimension
    // muliply dimension boxes together to get split element (or sum dim_splits
    // in the exponent)
    el_num_boxes_ptr[el_id] = pow(2, total_recursive_splits);
  });

  // Prefix sum to get offsets from number of splits
  int32 *offsets_ptr = offsets.get_device_ptr();
  RAJA::exclusive_scan<for_policy>(el_num_boxes_ptr, el_num_boxes_ptr + num_els,
    offsets_ptr, RAJA::operators::plus<int32>{});
  
  // Get total number of splits (last offset value plus the number of splits for
  // the last face)
  const int32 *offsets_host_ptr = offsets.get_host_ptr_const();
  const int32 *el_num_boxes_host_ptr = el_num_boxes.get_host_ptr_const();
  int32 total_boxes = offsets_host_ptr[num_els - 1] + el_num_boxes_host_ptr[num_els - 1];
  return total_boxes;
}


template <class ElemT>
BVH construct_wang_bvh (Mesh<ElemT> &mesh, Array<AABB<ElemT::get_dim ()>> &ref_aabbs) {
  DRAY_LOG_OPEN("construct_bvh");

  constexpr uint32 dim = ElemT::get_dim ();

  constexpr double bbox_scale = 1.000001;

  const int num_els = mesh.get_num_elem ();

  Array<AABB<>> aabbs;
  Array<int32> prim_ids;

  Array<int32> el_num_boxes;    // Number of boxes to be made for each element.
  Array<int32> el_splits_dim;   // Number of recursive splits to be made in each dimension.
  Array<int32> aabbs_offsets;   // Offsets for each element in output aabb array.
  int total_num_boxes;          // Total number of boxes, used for allocating space for aabbs.

  Timer wang_timer;
  total_num_boxes = detail::get_wang_recursive_splits(mesh, el_num_boxes, el_splits_dim, aabbs_offsets);
  DRAY_LOG_ENTRY("wang_calculations", wang_timer.elapsed());
  printf("Zone boxes calculated: total of %d boxes for %d elements in %d dimentions.\n", total_num_boxes, num_els, dim);

  aabbs.resize(total_num_boxes);
  prim_ids.resize(total_num_boxes);
  ref_aabbs.resize(total_num_boxes);

  AABB<> *aabb_ptr = aabbs.get_device_ptr();
  int32  *prim_ids_ptr = prim_ids.get_device_ptr();
  AABB<dim> *ref_aabbs_ptr = ref_aabbs.get_device_ptr();
  
  const int32 *el_num_boxes_ptr = el_num_boxes.get_device_ptr_const();
  const int32 *el_splits_dim_ptr = el_splits_dim.get_device_ptr_const();
  const int32 *aabbs_offsets_ptr = aabbs_offsets.get_device_ptr_const();

  DeviceMesh<ElemT> device_mesh (mesh);
  Timer timer;
  RAJA::forall<for_policy>(RAJA::RangeSegment(0, num_els), [=] DRAY_LAMBDA (int32 el_id)
  {
    const int32 num_boxes = el_num_boxes_ptr[el_id];

    // Could not dynaically allocate a new array because for small tolerances
    // we fail to find space sometimes.
    
    int32 box_idx = aabbs_offsets_ptr[el_id];

    // Get the bounds of the element in physical space
    // and initialize an initial reference space box.
    device_mesh.get_elem(el_id).get_bounds(aabb_ptr[box_idx]);
    ref_aabbs_ptr[box_idx] = AABB<dim>::ref_universe();

    int count = 1; // start off with single box in the array
    for(int d = 0; d < dim; ++d) {
      for(int split = el_splits_dim_ptr[(dim * el_id) + d]; split > 0; --split) {
        // Split each exisiting box along the appropriate dimension 
        int old_count = count;
        for(int box = 0; box < old_count; ++box) {
          int box_count_idx = box_idx + count;
          int split_boxs_idx = box_idx + box;
          // update reference bounds
          ref_aabbs_ptr[box_count_idx] = ref_aabbs_ptr[split_boxs_idx].split(d);
          // udpate the phys bounds
          device_mesh.get_elem(el_id).get_sub_bounds(ref_aabbs_ptr[split_boxs_idx],
                                                    aabb_ptr[split_boxs_idx]);
          device_mesh.get_elem(el_id).get_sub_bounds(ref_aabbs_ptr[box_count_idx],
                                                    aabb_ptr[box_count_idx]);
          count++;
        }
      }
    }

    for(int i = 0; i < num_boxes; ++i)
    {
      aabb_ptr[box_idx + i].scale(bbox_scale);
      prim_ids_ptr[box_idx + i] = el_id;
    }
  });
  DRAY_LOG_ENTRY("aabb_extraction", timer.elapsed());

  LinearBVHBuilder builder;
  BVH bvh = builder.construct(aabbs, prim_ids);
  DRAY_LOG_CLOSE();
  return bvh;
}


// get the flatness across each dimension.
template <uint32 dim, uint32 ncomp>
DRAY_EXEC Vec<Float, dim> get_flatness (Vec<Float, ncomp> *cntr_pts, int32 p_order) {
  // returns a metric of flatness
  const int32 stride_y = p_order + 1;
  const int32 stride_z = dim == 3 ? stride_y * stride_y : 0;
  const int32 strides [3] = {1, stride_y, stride_z};

  Vec<Float, dim> max_flatness;
  max_flatness.zero();

  for (int32 i = 0; i < p_order + 1; ++i) { 
    for (int32 j = 0; j < p_order + 1; ++j) {
      for (int32 d = 0; d < dim; d++) {
        Float current_flatness = 0;
        int32 outer_dim1 = (d + 1) % 3;
        int32 outer_dim2 = (d + 2) % 3;
        int32 start_idx = (strides[outer_dim1] * i) + (strides[outer_dim2] * j);
        // Calculate disntance between control points divided by distance between end points.
        // TODO: This might be a larger upper bound than wang's formula, find a better measure.
        for (int32 k = 0; k < p_order; ++k) {
          current_flatness += (cntr_pts[strides[d] * k + start_idx] +
              cntr_pts[strides[d] * (k + 1) + start_idx]).magnitude();
        }

        current_flatness /= (cntr_pts[start_idx] +
            cntr_pts[strides[d] * p_order + start_idx]).magnitude();

        if (current_flatness > max_flatness[d]) {
          max_flatness[d] = current_flatness;
        }
      }
    }
  }

  return max_flatness;
}


// TODO split this up to make less messy
template <class ElemT>
BVH construct_recursive_subdivision_bvh (Mesh<ElemT> &mesh, Array<AABB<ElemT::get_dim ()>> &ref_aabbs) {
  DRAY_LOG_OPEN("construct_bvh");

  constexpr uint32 dim = ElemT::get_dim ();
  constexpr uint32 ncomp = ElemT::get_ncomp ();
  const int32 p = mesh.get_poly_order();
  using PtrT = SharedDofPtr<Vec<Float, ncomp>>;

  constexpr double bbox_scale = 1.000001;

  const int num_els = mesh.get_num_elem ();

  const float32 flat_tol = dray::get_zone_flatness_tolerance();

  Array<AABB<>> aabbs;
  Array<int32> prim_ids;

  Array<int32> el_num_boxes;      // Number of boxes to be made for each element.
  Array<int32> el_splits_dim;     // Number of recursive splits to be made in each dimension.
  Array<int32> aabbs_offsets;     // Offsets for each element in output aabb array.
  
  int total_num_boxes;          // Total number of boxes, used for allocating space for aabbs.

  Timer wang_timer;
  total_num_boxes = detail::get_wang_recursive_splits(mesh, el_num_boxes, el_splits_dim, aabbs_offsets);
  DRAY_LOG_ENTRY("wang_calculations", wang_timer.elapsed());
  // printf("Zone boxes calculated: total of %d boxes for %d elements in %d dimentions.\n", total_num_boxes, num_els, dim);

  aabbs.resize(total_num_boxes);
  prim_ids.resize(total_num_boxes);
  ref_aabbs.resize(total_num_boxes);

  Array<int32> actual_num_boxes;  // Actual number of boxes created using recursive subdivision.
  actual_num_boxes.resize(num_els);
  Array<int32> actual_aabbs_offsets;  // offsets for elements boxes in the new array.
  actual_aabbs_offsets.resize(num_els);

  AABB<> *aabb_ptr = aabbs.get_device_ptr();
  int32  *prim_ids_ptr = prim_ids.get_device_ptr();
  AABB<dim> *ref_aabbs_ptr = ref_aabbs.get_device_ptr();
  int32  *act_num_boxes_ptr = actual_num_boxes.get_device_ptr();
  int32  *act_aabbs_offsets_ptr = actual_num_boxes.get_device_ptr();
  
  const int32 *el_num_boxes_ptr = el_num_boxes.get_device_ptr_const();
  const int32 *aabbs_offsets_ptr = aabbs_offsets.get_device_ptr_const();

  printf("aabb_ptr: %x \n", aabb_ptr);
  printf("prim_ids_ptr: %x \n", prim_ids_ptr);
  printf("ref_aabbs_ptr: %x \n", ref_aabbs_ptr);
  printf("act_num_boxes_ptr: %x \n", act_num_boxes_ptr);
  printf("act_aabbs_offsets_ptr: %x \n", act_aabbs_offsets_ptr);
  printf("act_aabbs_offsets_ptr: %x \n", act_aabbs_offsets_ptr);
  printf("el_num_boxes_ptr: %x \n", el_num_boxes_ptr);
  printf("aabbs_offsets_ptr: %x \n", aabbs_offsets_ptr);


  DeviceMesh<ElemT> device_mesh (mesh);
  Timer timer;
  RAJA::forall<for_policy>(RAJA::RangeSegment(0, num_els), [=] DRAY_LAMBDA (int32 el_id)
  {
    const int32 num_boxes = el_num_boxes_ptr[el_id];

    // Could not dynaically allocate a new array because for small tolerances
    // we fail to find space sometimes.
    int32 box_idx = aabbs_offsets_ptr[el_id];

    // Get the bounds of the element in physical space
    // and initialize an initial reference space box.
    device_mesh.get_elem(el_id).get_bounds(aabb_ptr[box_idx]);
    ref_aabbs_ptr[box_idx] = AABB<dim>::ref_universe();

    const int32 num_dofs = ElemT::get_num_dofs (p);
    Vec<Float, ncomp> *cntr_pts = new Vec<Float, ncomp>[num_dofs];


    int subdiv_idx = 0; // The index of the aabb which we are considering subdividing
                        // Every aabb before this index has been subdivided to meet the tolerance
    int count = 1;      // number of aabbs currently in the array
    while (subdiv_idx < count) { // iterate until all of the boxes are subdivided down
      int curr_idx = box_idx + subdiv_idx; // actual indicies
      int new_idx = box_idx + count;

      if (new_idx > num_boxes + box_idx) {
          printf("overstepped!");
          break;
      }
      // if flat enough over dimention d
      //    incremenet to the next dimension d
      // if not flat enough
      //    subdivide to get the right side, place it at the end of the array
      //    subdivide to get the left side, replace it with the box in subdiv_idx

      device_mesh.get_elem(el_id).get_sub_element(ref_aabbs_ptr[curr_idx], cntr_pts);
      Vec<Float, dim> flatness = detail::get_flatness<dim, ncomp> (cntr_pts, p);
      if (flatness.max() > flat_tol) {

        // Pick the smallest dimention with the smallest flatness that still
        // does not meet the tolerance. We do this to minimize the number of
        // splits across the dimensions.
        int32 min_dim = 0;
        for (int32 d = 0; d < dim; ++d) {
          if (flatness[min_dim] > flatness[d] && flatness[d] > flat_tol)
            min_dim = d;
        }

        // box operations
        // update reference bounds
        ref_aabbs_ptr[new_idx] = ref_aabbs_ptr[curr_idx].split(min_dim);
        // udpate the phys bounds
        device_mesh.get_elem(el_id).get_sub_bounds(ref_aabbs_ptr[curr_idx],
                                                  aabb_ptr[curr_idx]);
        device_mesh.get_elem(el_id).get_sub_bounds(ref_aabbs_ptr[new_idx],
                                                  aabb_ptr[new_idx]);
        ++count;
      }
      else{
        ++subdiv_idx;
      }
    }

    delete[] cntr_pts;

    for(int i = 0; i < count; ++i)
    {
      aabb_ptr[box_idx + i].scale(bbox_scale);
      prim_ids_ptr[box_idx + i] = el_id;
    }

    act_num_boxes_ptr[el_id] = count;
  });

  // Afterwards, shorten the length of the aabb and prim_id arrays.
  RAJA::exclusive_scan<for_policy> (act_num_boxes_ptr, act_num_boxes_ptr + num_els,
      act_aabbs_offsets_ptr, RAJA::operators::plus<int32>{});

  // Fill in the new arrays with no empty cells.
  Array<AABB<>> actual_aabbs;
  Array<int32> actual_prim_ids;
  int32 compressed_num_boxes = actual_aabbs_offsets.get_value(num_els - 1) + actual_num_boxes.get_value(num_els - 1);
  actual_aabbs.resize(compressed_num_boxes);
  actual_prim_ids.resize(compressed_num_boxes);

  AABB<> *act_aabb_ptr = actual_aabbs.get_device_ptr();
  int32  *act_prim_ids_ptr = actual_prim_ids.get_device_ptr();

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, num_els), [=] DRAY_LAMBDA (int32 el_id)
  {
    int32 box_idx = aabbs_offsets_ptr[el_id];
    int32 new_box_idx = act_aabbs_offsets_ptr[el_id];
    int32 act_n_boxes = act_num_boxes_ptr[el_id];

    for (int32 i = 0; i < act_n_boxes; ++i) {
      act_aabb_ptr[new_box_idx + i] = aabb_ptr[box_idx + i];
      act_prim_ids_ptr[new_box_idx + i] = prim_ids_ptr[box_idx + i];
    }
  });

  DRAY_LOG_ENTRY("aabb_extraction", timer.elapsed());

  LinearBVHBuilder builder;
  BVH bvh = builder.construct(actual_aabbs, actual_prim_ids);
  DRAY_LOG_CLOSE();
  return bvh;
}

} // namespace detail

} // namespace dray


//
// Explicit instantiations.
//
namespace dray
{
namespace detail
{
//
// reorder();
//
template void reorder (Array<int32> &indices, Array<float32> &array);
template void reorder (Array<int32> &indices, Array<float64> &array);


//
// extract_faces();   // Quad
//
template Array<Vec<int32, 4>>
extract_faces (Mesh<MeshElem<3u, ElemType::Quad, Order::General>> &mesh);

//
// extract_faces();   // Tri
//
/// template
/// Array<Vec<int32,4>> extract_faces(Mesh<float32, MeshElem<float32, 3u, ElemType::Tri, Order::General>> &mesh);
/// template
/// Array<Vec<int32,4>> extract_faces(Mesh<float64, MeshElem<float64, 3u, ElemType::Tri, Order::General>> &mesh);


//
// construct_bvh();   // Quad
//
template BVH construct_bvh (Mesh<MeshElem<2u, ElemType::Quad, Order::General>> &mesh,
                            Array<AABB<2>> &ref_aabbs);
template BVH construct_bvh (Mesh<MeshElem<3u, ElemType::Quad, Order::General>> &mesh,
                            Array<AABB<3>> &ref_aabbs);
template BVH construct_bvh (Mesh<MeshElem<3u, ElemType::Quad, Order::Linear>> &mesh,
                            Array<AABB<3>> &ref_aabbs);

//
// construct_bvh();   // Tri
//
/// template
/// BVH construct_bvh(Mesh<float32, MeshElem<float32, 2u, ElemType::Tri, Order::General>> &mesh, Array<AABB<2>> &ref_aabbs);
/// template
/// BVH construct_bvh(Mesh<float32, MeshElem<float32, 3u, ElemType::Tri, Order::General>> &mesh, Array<AABB<3>> &ref_aabbs);
/// template
/// BVH construct_bvh(Mesh<float64, MeshElem<float64, 2u, ElemType::Tri, Order::General>> &mesh, Array<AABB<2>> &ref_aabbs);
/// template
/// BVH construct_bvh(Mesh<float64, MeshElem<float64, 3u, ElemType::Tri, Order::General>> &mesh, Array<AABB<3>> &ref_aabbs);

//
// construct_wang_bvh();   // Quad
//
template BVH construct_wang_bvh (Mesh<MeshElem<2u, ElemType::Quad, Order::General>> &mesh,
                            Array<AABB<2>> &ref_aabbs);
template BVH construct_wang_bvh (Mesh<MeshElem<3u, ElemType::Quad, Order::General>> &mesh,
                            Array<AABB<3>> &ref_aabbs);
template BVH construct_wang_bvh (Mesh<MeshElem<3u, ElemType::Quad, Order::Linear>> &mesh,
                            Array<AABB<3>> &ref_aabbs);
template BVH construct_recursive_subdivision_bvh (Mesh<MeshElem<2u, ElemType::Quad, Order::General>> &mesh,
                            Array<AABB<2>> &ref_aabbs);
template BVH construct_recursive_subdivision_bvh (Mesh<MeshElem<3u, ElemType::Quad, Order::General>> &mesh,
                            Array<AABB<3>> &ref_aabbs);
template BVH construct_recursive_subdivision_bvh (Mesh<MeshElem<3u, ElemType::Quad, Order::Linear>> &mesh,
                            Array<AABB<3>> &ref_aabbs);

} // namespace detail
} // namespace dray
