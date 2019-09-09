#ifndef DRAY_MESH_UTILS_HPP
#define DRAY_MESH_UTILS_HPP

#include <dray/GridFunction/mesh.hpp>

namespace dray
{

namespace detail
{

  DRAY_EXEC void swap(int32 &a, int32 &b);
  
  DRAY_EXEC void sort4(Vec<int32,4> &vec);
  
  template<typename T>
  void reorder(Array<int32> &indices, Array<T> &array);
  
  Array<int32> sort_faces(Array<Vec<int32,4>> &faces);
  
  DRAY_EXEC bool is_same(const Vec<int32,4> &a, const Vec<int32,4> &b);
  
  void unique_faces(Array<Vec<int32,4>> &faces, Array<int32> &orig_ids);
  
  // Returns 6 faces for each element, each face
  // represented by the ids of the corner dofs.
  //TODO extract_faces() needs to be extended to triangular/tetrahedral meshes too.
  template<typename T, class ElemT>
  Array<Vec<int32,4>> extract_faces(Mesh<T, ElemT> &mesh);
  
  // Returns faces, where faces[i][0] = el_id and 0 <= faces[i][1] = face_id < 6.
  // This allows us to identify the needed dofs for a face mesh.
  Array<Vec<int32,2>> reconstruct(Array<int32> &orig_ids);
  
  //TODO
  /// template<typename T, class ElemT>
  /// BVH construct_face_bvh(Mesh<T, ElemT> &mesh, Array<Vec<int32,2>> &faces);
  
  //TODO
  /// template<typename T, class ElemT>
  /// typename Mesh<T, ElemT>::ExternalFaces  external_faces(Mesh<T, ElemT> &mesh);
  
  template<typename T, class ElemT>
  BVH construct_bvh(Mesh<T, ElemT> &mesh, Array<AABB<ElemT::get_dim()>> &ref_aabbs);

}//namespace detail

}//namespace dray


#endif//DRAY_MESH_UTILS_HPP
