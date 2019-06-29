#include <dray/GridFunction/mesh.hpp>
#include <dray/aabb.hpp>
#include <RAJA/RAJA.hpp>
#include <dray/policies.hpp>

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
      device_mesh.get_elem(el_id).get_sub_bounds(ref_boxs[max_id].m_ranges, boxs[max_id].m_ranges);
      device_mesh.get_elem(el_id).get_sub_bounds(ref_boxs[count].m_ranges, boxs[count].m_ranges);
      count++;
    }

    AABB<> res;
    for(int i = 0; i < splits; ++i)
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
BVH Mesh<T,dim>::get_bvh()
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

// Explicit instantiations.
template class MeshAccess<float32, 3>;
template class MeshAccess<float64, 3>;

// Explicit instantiations.
template class Mesh<float32, 3>;
template class Mesh<float64, 3>;
}
