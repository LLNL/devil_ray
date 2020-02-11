#include <dray/rendering/contour.hpp>

#include <dray/array_utils.hpp>
#include <dray/error.hpp>
#include <dray/dispatcher.hpp>

#include <dray/isosurface_intersection.hpp>
#include <dray/GridFunction/device_mesh.hpp>
#include <dray/GridFunction/device_field.hpp>
#include <dray/utils/data_logger.hpp>

#include <assert.h>

namespace dray
{
namespace detail
{

DRAY_EXEC_ONLY
bool intersect_AABB(const Vec<float32,4> *bvh,
                    const int32 &currentNode,
                    const Vec<Float,3> &orig_dir,
                    const Vec<Float,3> &inv_dir,
                    const Float& closest_dist,
                    bool &hit_left,
                    bool &hit_right,
                    const Float &min_dist) //Find hit after this distance
{
  Vec<float32, 4> first4  = const_get_vec4f(&bvh[currentNode + 0]);
  Vec<float32, 4> second4 = const_get_vec4f(&bvh[currentNode + 1]);
  Vec<float32, 4> third4  = const_get_vec4f(&bvh[currentNode + 2]);
  Float xmin0 = first4[0] * inv_dir[0] - orig_dir[0];
  Float ymin0 = first4[1] * inv_dir[1] - orig_dir[1];
  Float zmin0 = first4[2] * inv_dir[2] - orig_dir[2];
  Float xmax0 = first4[3] * inv_dir[0] - orig_dir[0];
  Float ymax0 = second4[0] * inv_dir[1] - orig_dir[1];
  Float zmax0 = second4[1] * inv_dir[2] - orig_dir[2];
  Float min0 = fmaxf(
    fmaxf(fmaxf(fminf(ymin0, ymax0), fminf(xmin0, xmax0)), fminf(zmin0, zmax0)),
    min_dist);
  Float max0 = fminf(
    fminf(fminf(fmaxf(ymin0, ymax0), fmaxf(xmin0, xmax0)), fmaxf(zmin0, zmax0)),
    closest_dist);
  hit_left = (max0 >= min0);

  Float xmin1 = second4[2] * inv_dir[0] - orig_dir[0];
  Float ymin1 = second4[3] * inv_dir[1] - orig_dir[1];
  Float zmin1 = third4[0] * inv_dir[2] - orig_dir[2];
  Float xmax1 = third4[1] * inv_dir[0] - orig_dir[0];
  Float ymax1 = third4[2] * inv_dir[1] - orig_dir[1];
  Float zmax1 = third4[3] * inv_dir[2] - orig_dir[2];

  Float min1 = fmaxf(
    fmaxf(fmaxf(fminf(ymin1, ymax1), fminf(xmin1, xmax1)), fminf(zmin1, zmax1)),
    min_dist);
  Float max1 = fminf(
    fminf(fminf(fmaxf(ymin1, ymax1), fmaxf(xmin1, xmax1)), fmaxf(zmin1, zmax1)),
    closest_dist);
  hit_right = (max1 >= min1);

  return (min0 > min1);
}

void init_hits(Array<RayHit> &hits)
{
  const int32 size = hits.size();
  RayHit * hits_ptr = hits.get_device_ptr();
  Vec<Float,3> the_ninety_nine = {-99, 99, -99};

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, size), [=] DRAY_LAMBDA (const int32 i)
  {
    RayHit hit;
    hit.m_hit_idx = -1;
    hit.m_ref_pt = the_ninety_nine;
    hits_ptr[i] = hit;
  });
}

template<class ElemT>
struct ContourIntersector
{
  DeviceMesh<ElemT> m_device_mesh;
  DeviceField<FieldOn<ElemT, 1u>> m_device_field;
  const Float m_iso_val;

  ContourIntersector(DeviceMesh<ElemT> &device_mesh,
                     DeviceField<FieldOn<ElemT, 1u>> &device_field,
                     const Float &iso_val)
   : m_device_mesh(device_mesh),
     m_device_field(device_field),
     m_iso_val(iso_val)

  {}


  DRAY_EXEC RayHit intersect_contour(const Ray &ray,
                                     const int32 &el_idx,
                                     const AABB<3> &ref_box,
                                     stats::Stats &mstat) const
  {
    const bool use_init_guess = true;
    RayHit hit;
    hit.m_hit_idx = -1;

    // Alternatives: we could just subelement range
    AABB<1u> aabb_range;
    m_device_field.get_elem(el_idx).get_sub_bounds(ref_box, aabb_range);
    Range range = aabb_range.m_ranges[0];
    if(m_iso_val >= range.min() && m_iso_val <= range.max())
    {
      mstat.acc_candidates(1);
      hit.m_ref_pt = ref_box.template center<Float> ();
      hit.m_dist = ray.m_near;

      bool inside = Intersector_RayIsosurf<ElemT>::intersect_local (mstat,
                                                 m_device_mesh.get_elem(el_idx),
                                                 m_device_field.get_elem(el_idx),
                                                 ray,
                                                 m_iso_val,
                                                 hit.m_ref_pt,// initial ref guess
                                                 hit.m_dist,  // initial ray guess
                                                 use_init_guess);
      //bool inside =
      //  Intersector_RayIsosurf<ElemT>::intersect(mstat,
      //                                           m_device_mesh.get_elem(el_idx),
      //                                           m_device_field.get_elem(el_idx),
      //                                           ray,
      //                                           m_iso_val,
      //                                           ref_box,
      //                                           hit.m_ref_pt,
      //                                           hit.m_dist,
      //                                           use_init_guess);
      if(inside)
      {
        hit.m_hit_idx = el_idx;
      }
    }
    return hit;
  }

};

template <class ElemT>
void
intersect_isosurface(const Array<Ray> &rays,
                     const float32 &iso_val,
                     Field<FieldOn<ElemT, 1u>> &field,
                     Mesh<ElemT> &mesh,
                     Array<RayHit> &hits)
{
  // This method intersects rays with the isosurface using the Newton-Raphson method.
  // The system of equations to be solved is composed from
  //   ** Transformations **
  //   1. PHI(u,v,w)  -- mesh element transformation, from ref space to R3.
  //   2. F(u,v,w)    -- scalar field element transformation, from ref space to R1.
  //   3. r(s)        -- ray parameterized by distance, relative to ray origin.
  //                     (We only restrict s >= 0. No expectation of s <= 1.)
  //   ** Targets **
  //   4. F_0         -- isovalue.
  //   5. Orig        -- ray origin.
  //
  // The ray-isosurface intersection is a solution to the following system:
  //
  // [ [PHI(u,v,w)]   [r(s)]         [ [      ]
  //   [          ] - [    ]     ==    [ Orig ]
  //   [          ]   [    ]           [      ]
  //   F(u,v,w)     +   0    ]           F_0    ]

  // Initialize outputs.
  //init_hits(hits); // TODO: not clear why we can't init inside main function

  const Vec<Float,3> element_guess = {0.5, 0.5, 0.5};
  const Float ray_guess = 1.0;

  // things we need fromt the bvh
  BVH bvh = mesh.get_bvh();
  const int32 *leaf_ptr = bvh.m_leaf_nodes.get_device_ptr_const();
  const Vec<float32, 4> *inner_ptr = bvh.m_inner_nodes.get_device_ptr_const();
  const int32 *aabb_ids_ptr = bvh.m_aabb_ids.get_device_ptr_const();
  const AABB<3> *ref_aabb_ptr = mesh.get_ref_aabbs().get_device_ptr_const();

  const int32 size = rays.size();

  DeviceMesh<ElemT> device_mesh(mesh);
  DeviceField<FieldOn<ElemT, 1u>> device_field(field);
  ContourIntersector<ElemT> intersector(device_mesh, device_field, iso_val);

  const Ray *ray_ptr = rays.get_device_ptr_const();

  RayHit *hit_ptr = hits.get_device_ptr();

  Array<stats::Stats> mstats;
  mstats.resize(size);
  stats::Stats *mstats_ptr = mstats.get_device_ptr();

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, size), [=] DRAY_LAMBDA (int32 i)
  {

    const Ray &ray = ray_ptr[i];
    RayHit hit;
    hit.m_hit_idx = -1;

    stats::Stats mstat;
    mstat.construct();

    Float closest_dist = ray.m_far;
    Float min_dist = ray.m_near;
    ///int32 hit_idx = -1;
    const Vec<Float,3> dir = ray.m_dir;
    Vec<Float,3> inv_dir;
    inv_dir[0] = rcp_safe(dir[0]);
    inv_dir[1] = rcp_safe(dir[1]);
    inv_dir[2] = rcp_safe(dir[2]);

    int32 current_node;
    int32 todo[64];
    int32 stackptr = 0;
    current_node = 0;

    constexpr int32 barrier = -2000000000;
    todo[stackptr] = barrier;

    const Vec<Float,3> orig = ray.m_orig;

    Vec<Float,3> orig_dir;
    orig_dir[0] = orig[0] * inv_dir[0];
    orig_dir[1] = orig[1] * inv_dir[1];
    orig_dir[2] = orig[2] * inv_dir[2];

    while (current_node != barrier)
    {
      if (current_node > -1)
      {
        bool hit_left, hit_right;
        bool right_closer = intersect_AABB(inner_ptr,
                                           current_node,
                                           orig_dir,
                                           inv_dir,
                                           closest_dist,
                                           hit_left,
                                           hit_right,
                                           min_dist);

        if (!hit_left && !hit_right)
        {
          current_node = todo[stackptr];
          stackptr--;
        }
        else
        {
          Vec<float32, 4> children = const_get_vec4f(&inner_ptr[current_node + 3]);
          int32 l_child;
          constexpr int32 isize = sizeof(int32);
          memcpy(&l_child, &children[0], isize);
          int32 r_child;
          memcpy(&r_child, &children[1], isize);
          current_node = (hit_left) ? l_child : r_child;

          if (hit_left && hit_right)
          {
            if (right_closer)
            {
              current_node = r_child;
              stackptr++;
              todo[stackptr] = l_child;
            }
            else
            {
              stackptr++;
              todo[stackptr] = r_child;
            }
          }
        }
      } // if inner node

      if (current_node < 0 && current_node != barrier) //check register usage
      {
        current_node = -current_node - 1; //swap the neg address

        // we might want to figure out how to short circuit early here by checking
        // distance
        int32 el_idx = leaf_ptr[current_node];
        const AABB<3> ref_box = ref_aabb_ptr[aabb_ids_ptr[current_node]];
        RayHit el_hit = intersector.intersect_contour(ray, el_idx, ref_box, mstat);

        if(el_hit.m_hit_idx != -1 && el_hit.m_dist < closest_dist && el_hit.m_dist > min_dist)
        {
          hit = el_hit;
          closest_dist = hit.m_dist;
          mstat.found();
        }

        current_node = todo[stackptr];
        stackptr--;
      } // if leaf node

    } //while

    mstats_ptr[i] = mstat;
    hit_ptr[i] = hit;

  });

  stats::StatStore::add_ray_stats(rays, mstats);
}

struct ContourFunctor
{
  Contour *m_iso;
  Array<Ray> *m_rays;
  Array<RayHit> m_hits;
  ContourFunctor(Contour *iso,
                 Array<Ray> *rays)
    : m_iso(iso),
      m_rays(rays)
  {
  }

  template<typename TopologyType, typename FieldType>
  void operator()(TopologyType &topo, FieldType &field)
  {
    m_hits = m_iso->execute(topo.mesh(), field, *m_rays);
  }
};

} // namespace detail

Contour::Contour(DataSet &data_set)
  : Traceable(data_set),
    m_iso_value(infinity32())
{
}

Contour::~Contour()
{
}

Array<RayHit>
Contour::nearest_hit(Array<Ray> &rays)
{
  assert(m_iso_field_name != "");

  TopologyBase *topo = m_data_set.topology();
  FieldBase *field = m_data_set.field(m_iso_field_name);

  detail::ContourFunctor func(this, &rays);
  dispatch_3d(topo, field, func);
  return func.m_hits;
}

template<class MeshElement, class FieldElement>
Array<RayHit>
Contour::execute(Mesh<MeshElement> &mesh,
                 Field<FieldElement> &field,
                 Array<Ray> &rays)
{
  DRAY_LOG_OPEN("isosuface");

  if(m_iso_value == infinity32())
  {
    DRAY_ERROR("Contour: no iso value set");
  }

  const int32 num_elems = mesh.get_num_elem();

  Array<RayHit> hits;
  hits.resize(rays.size());

  // Intersect rays with isosurface.
  detail::intersect_isosurface(rays,
                               m_iso_value,
                               field,
                               mesh,
                               hits);

  DRAY_LOG_CLOSE();
  return hits;
}

void
Contour::iso_field(const std::string field_name)
{
 m_iso_field_name = field_name;
}

void
Contour::iso_value(const float32 iso_value)
{
  m_iso_value = iso_value;
}

}//namespace dray

