#include <dray/filters/mesh_boundary.hpp>

#include <dray/data_set.hpp>
#include <dray/GridFunction/mesh.hpp>
#include <dray/GridFunction/mesh_utils.hpp>


namespace dray
{

  template<typename T, class ElemT>
  DataSet<T, NDElem<ElemT, 2>> MeshBoundary::execute(DataSet<T, ElemT> &data_set)
  {
    // copy_face_dof_subset
    // - If true, make a new array filled with just the surface geometry
    //   The dof ids for the surface mesh will no longer correspond to
    //   their ids in the volume mesh.
    // - If false, the old geometry array will be re-used, so dof ids
    //   will match the volume mesh.
    const bool copy_face_dof_subset = false;

    using Elem3D = ElemT;
    using Elem2D = NDElem<ElemT, 2>;

    Mesh<T, ElemT> orig_mesh = data_set.get_mesh();
    const int32 mesh_poly_order = orig_mesh.get_poly_order();

    // Step 1: Extract the boundary mesh: Matt's external_faces() algorithm.

    // Identify unique/external faces.
    Array<Vec<int32,4>> face_corner_ids = detail::extract_faces(orig_mesh);
    Array<int32> orig_face_idx = detail::sort_faces(face_corner_ids);
    detail::unique_faces(face_corner_ids, orig_face_idx);
    Array<Vec<int32, 2>> elid_faceid = detail::reconstruct(orig_face_idx);

    // Copy the dofs for each face.
    GridFunctionData<T, 3u> mesh_data_2d;   // The 3u means 3 components (embedded in 3D).

    if (copy_face_dof_subset)    // New geometry array with subset of dofs.
    {
      //TODO
    }
    else                         // Re-use the old geometry array.
    {
      // Make sure to initialize all 5 members. TODO a new constructor?
      GridFunctionData<T, 3u> orig_mesh_data_3d = orig_mesh.get_dof_data();

      mesh_data_2d.m_el_dofs = (mesh_poly_order+1)*(mesh_poly_order+1);
      mesh_data_2d.m_size_el = elid_faceid.size();
      mesh_data_2d.m_size_ctrl = orig_mesh_data_3d.m_size_ctrl;
      mesh_data_2d.m_values = orig_mesh_data_3d.m_values;

      mesh_data_2d.m_ctrl_idx.resize(elid_faceid.size());

      const Vec<int32,2> * elid_faceid_ptr = elid_faceid.get_device_ptr_const();
      const int32 * orig_dof_idx_ptr       = orig_mesh_data_3d.m_ctrl_idx.get_device_ptr_const();
      int32 * new_dof_idx_ptr              = mesh_data_2d.m_ctrl_idx.get_device_ptr();

      RAJA::forall<for_policy>(RAJA::RangeSegment(0, mesh_data_2d.m_size_el), [=] DRAY_LAMBDA (int32 face_idx)
      {
        const int32 eldofs0 = 1;
        const int32 eldofs1 = eldofs0 * (mesh_poly_order+1);
        const int32 eldofs2 = eldofs1 * (mesh_poly_order+1);
        const int32 eldofs3 = eldofs2 * (mesh_poly_order+1);
        const int32 axis_strides[3] = {eldofs0, eldofs1, eldofs2};

        const int32 faceid      = elid_faceid_ptr[face_idx][1];
        const int32 orig_offset = elid_faceid_ptr[face_idx][0] * eldofs3;
        const int32 new_offset  = face_idx * eldofs2;

        const int32 face_axis = (faceid == 0 || faceid == 3 ? 0
                               : faceid == 1 || faceid == 4 ? 1
                                                            : 2);

        const int32 face_start = (faceid < 3 ? 0 : (eldofs1 - 1) * axis_strides[face_axis]);
        const int32 major_stride = (face_axis == 2 ? 1 : 2);
        const int32 minor_stride = (face_axis == 0 ? 1 : 0);

        for (int32 ii = 0; ii < eldofs1; ii++)
          for (int32 jj = 0; jj < eldofs1; jj++)
            new_dof_idx_ptr[new_offset + eldofs1*ii + jj] =
                orig_dof_idx_ptr[orig_offset + face_start + major_stride*ii + minor_stride*jj];
      });
    }

    // Wrap the mesh data inside a mesh and dataset.
    Mesh<T, Elem2D> boundary_mesh(mesh_data_2d, mesh_poly_order);
    DataSet<T, Elem2D> boundary_dataset(boundary_mesh);

    // Step 2: For each field, add boundary field to the boundary_dataset.
    //TODO

    return boundary_dataset;
  }



  // Explicit instantiations.
  //

  // <float32, Quad>
  template
    DataSet<float32, MeshElem<float32, 2u, ElemType::Quad, Order::General>>
    MeshBoundary::execute<float32, MeshElem<float32, 3u, ElemType::Quad, Order::General>>(
        DataSet<float32, MeshElem<float32, 3u, ElemType::Quad, Order::General>> &data_set);

  // <float32, Tri>
  /// template
  ///   DataSet<float32, MeshElem<float32, 2u, ElemType::Tri, Order::General>>
  ///   MeshBoundary::execute<float32, MeshElem<float32, 3u, ElemType::Tri, Order::General>>(
  ///       DataSet<float32, MeshElem<float32, 3u, ElemType::Tri, Order::General>> &data_set);

  // <float64, Quad>
  template
    DataSet<float64, MeshElem<float64, 2u, ElemType::Quad, Order::General>>
    MeshBoundary::execute<float64, MeshElem<float64, 3u, ElemType::Quad, Order::General>>(
        DataSet<float64, MeshElem<float64, 3u, ElemType::Quad, Order::General>> &data_set);

  // <float64, Tri>
  /// template
  ///   DataSet<float64, MeshElem<float64, 2u, ElemType::Tri, Order::General>>
  ///   MeshBoundary::execute<float64, MeshElem<float64, 3u, ElemType::Tri, Order::General>>(
  ///       DataSet<float64, MeshElem<float64, 3u, ElemType::Tri, Order::General>> &data_set);

}//namespace dray
