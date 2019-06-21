#include <dray/GridFunction/mesh.hpp>
#include <dray/aabb.hpp>
#include <RAJA/RAJA.hpp>
#include <dray/policies.hpp>

namespace dray
{
  template <typename T, int32 dim>
  Array<AABB<dim>> Mesh<T,dim>::get_aabbs() const
  {
    constexpr double bbox_scale = 1.000001;

    Array<AABB<dim>> bboxes;
    bboxes.resize(get_num_elem());

    AABB<dim> *bboxes_ptr = bboxes.get_device_ptr();
    MeshAccess<T,dim> device_mesh = access_device_mesh();

    RAJA::forall<for_policy>(RAJA::RangeSegment(0, get_num_elem()), [=] DRAY_LAMBDA (int32 ii)
    {
      AABB<dim> bbox;
      device_mesh.get_elem(ii).get_bounds(bbox);
      bbox.scale(bbox_scale);    // Slightly scale the bbox to account for numerical noise
      bboxes_ptr[ii] = bbox;
    });

    return bboxes;
  }
 
  // Explicit instantiations.
  template class MeshAccess<float32, 3>;
  template class MeshAccess<float64, 3>;

  // Explicit instantiations.
  template class Mesh<float32, 3>;
  template class Mesh<float64, 3>;
}
