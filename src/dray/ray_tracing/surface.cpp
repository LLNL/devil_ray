#include <dray/ray_tracing/surface.hpp>

#include <dray/GridFunction/device_mesh.hpp>
#include <dray/error.hpp>
#include <dray/array_utils.hpp>
#include <dray/dispatcher.hpp>
#include <dray/ref_point.hpp>
#include <dray/device_framebuffer.hpp>
#include <dray/shaders.hpp>
#include <dray/face_intersection.hpp>
#include <dray/utils/data_logger.hpp>


namespace dray
{
namespace ray_tracing
{
namespace detail
{

//
// intersect_AABB()
//
// Copied verbatim from triangle_mesh.cpp
//
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

struct Candidates
{
 Array<int32> m_candidates;
 Array<int32> m_ref_aabb_ids;
};
//
// candidate_ray_intersection()
//
//   TODO find appropriate place for this function. It is mostly copied from TriangleMesh
//
template <int32 max_candidates>
Candidates candidate_ray_intersection(Array<Ray> rays, const BVH bvh)
{
  const int32 size = rays.size();


  Array<int32> candidates;
  candidates.resize(size * max_candidates);
  array_memset(candidates, -1);

  Array<int32> ref_aabb_ids;
  ref_aabb_ids.resize(size * max_candidates);


  const Ray *ray_ptr = rays.get_device_ptr_const();

  const int32 *leaf_ptr = bvh.m_leaf_nodes.get_device_ptr_const();
  const int32 *aabb_ids_ptr = bvh.m_aabb_ids.get_device_ptr_const();
  const Vec<float32, 4> *inner_ptr = bvh.m_inner_nodes.get_device_ptr_const();

  int32 *candidates_ptr = candidates.get_device_ptr();
  int32 *ref_aabb_ids_ptr = ref_aabb_ids.get_device_ptr();

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, size), [=] DRAY_LAMBDA (int32 i)
  {

    const Ray &ray = ray_ptr[i];

    Float closest_dist = ray.m_far;
    Float min_dist = ray.m_near;
    ///int32 hit_idx = -1;
    const Vec<Float,3> dir = ray.m_dir;
    Vec<Float,3> inv_dir;
    inv_dir[0] = rcp_safe(dir[0]);
    inv_dir[1] = rcp_safe(dir[1]);
    inv_dir[2] = rcp_safe(dir[2]);

    int32 current_node;
    int32 todo[max_candidates];
    int32 stackptr = 0;
    current_node = 0;

    constexpr int32 barrier = -2000000000;
    todo[stackptr] = barrier;

    const Vec<Float,3> orig = ray.m_orig;

    Vec<Float,3> orig_dir;
    orig_dir[0] = orig[0] * inv_dir[0];
    orig_dir[1] = orig[1] * inv_dir[1];
    orig_dir[2] = orig[2] * inv_dir[2];

    int32 candidate_idx = 0;

    while (current_node != barrier && candidate_idx < max_candidates)
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

        // Any leaf bbox we enter is a candidate.
        candidates_ptr[candidate_idx + i * max_candidates] = leaf_ptr[current_node];
        ref_aabb_ids_ptr[i * max_candidates + candidate_idx] = aabb_ids_ptr[current_node];
        candidate_idx++;

        current_node = todo[stackptr];
        stackptr--;
      } // if leaf node

    } //while

  });

  Candidates i_candidates;
  i_candidates.m_candidates = candidates;
  i_candidates.m_ref_aabb_ids = ref_aabb_ids;

  return i_candidates;
}

struct HasCandidate
{
  int32 m_max_candidates;
  const int32 *m_candidates_ptr;
  DRAY_EXEC bool operator() (int32 ii) const
  {
    return (m_candidates_ptr[ii * m_max_candidates] > -1);
  }
};

}  // namespace detail


Surface::Surface(DataSet &dataset)
  : Traceable(dataset)
{
}

Surface::~Surface()
{
}

template <typename ElemT>
Array<RayHit> intersect_mesh_faces(const Array<Ray> rays, const Mesh<ElemT> &mesh)
{
  if (ElemT::get_dim() != 2)
  {
    DRAY_ERROR("Cannot intersect_mesh_faces() on a non-surface mesh. "
                    "(Do you need the MeshBoundary filter?)");
  }

  constexpr int32 ref_dim = 2;

  // Initialize rpoints to same size as rays, each rpoint set to invalid_refpt.
  Array<RayHit> hits;
  hits.resize(rays.size());

  // Get intersection candidates for all active rays.
  constexpr int32 max_candidates = 100;
  detail::Candidates candidates =
      detail::candidate_ray_intersection<max_candidates> (rays, mesh.get_bvh());

  const AABB<ref_dim> *ref_aabbs_ptr = mesh.get_ref_aabbs().get_device_ptr_const();
  const int32 *candidates_ptr = candidates.m_candidates.get_device_ptr_const();
  const int32 *ref_aabb_ids_ptr = candidates.m_ref_aabb_ids.get_device_ptr_const();

  const int32 size = rays.size();

    // Define pointers for RAJA kernel.
  DeviceMesh<ElemT> device_mesh(mesh);
  const Ray *ray_ptr = rays.get_device_ptr_const();
  RayHit *hit_ptr = hits.get_device_ptr();

  Array<stats::Stats> mstats;
  mstats.resize(size);
  stats::Stats *mstats_ptr = mstats.get_device_ptr();

  // For each active ray, loop through candidates until found an intersection.
  RAJA::forall<for_policy>(RAJA::RangeSegment(0, size), [=] DRAY_LAMBDA (const int32 i)
  {
    stats::Stats mstat;
    mstat.construct();
    // Outputs to arrays.
    const Ray ray = ray_ptr[i];
    RayHit hit{ -1, infinity<Float>(),{-1.f,-1.f,-1.f} };

    // Local results.
    Vec<Float, ref_dim> ref_coords;

    // In case no intersection is found.
    // TODO change comparisons for valid rays to check both near and far.
    Float dist;

    bool found_any = false;
    int32 candidate_idx = 0;
    int32 candidate = candidates_ptr[i*max_candidates + candidate_idx];
    while (candidate_idx < max_candidates && candidate != -1)
    {
      bool found_inside = false;

      // Get candidate face.
      const int32 elid = candidate;
      const ElemT surf_elem = device_mesh.get_elem(elid);
      const int32 ref_id = ref_aabb_ids_ptr[i * max_candidates + candidate_idx];
      const AABB<ref_dim> ref_box_start = ref_aabbs_ptr[ref_id];

      const bool use_init_guess = true;
      mstat.acc_candidates(1);

      found_inside = Intersector_RayFace<ElemT>::intersect(mstat,
                                                           surf_elem,
                                                           ray,
                                                           ref_box_start,
                                                           ref_coords,
                                                           dist,
                                                           use_init_guess);

      if (found_inside && dist < ray.m_far && dist > ray.m_near && dist < hit.m_dist)
      {
        found_any = true;
        // Save the candidate result.
        hit.m_hit_idx = candidate;
        hit.m_ref_pt[0] = ref_coords[0];
        hit.m_ref_pt[1] = ref_coords[1];
        hit.m_dist = dist;
        mstat.found();
      }

      // Continue searching with the next candidate.
      candidate_idx++;
      candidate = candidates_ptr[i*max_candidates + candidate_idx];

    } // end while

    mstats_ptr[i] = mstat;
    hit_ptr[i] = hit;
  });  // end RAJA

  stats::StatStore::add_ray_stats(rays, mstats);
  return hits;
}

struct Functor
{
  Surface *m_lines;
  Array<Ray> *m_rays;
  Array<RayHit> m_hits;

  Functor(Surface *lines,
          Array<Ray> *rays)
    : m_lines(lines),
      m_rays(rays)
  {
  }

  template<typename TopologyType>
  void operator()(TopologyType &topo)
  {
    m_hits = m_lines->execute(topo.mesh(), *m_rays);
  }
};

Array<RayHit>
Surface::nearest_hit(Array<Ray> &rays)
{
  TopologyBase *topo = m_data_set.topology();

  Functor func(this, &rays);
  dispatch_2d(topo, func);
  return func.m_hits;
}

template<typename MeshElem>
Array<RayHit>
Surface::execute(Mesh<MeshElem> &mesh,
                 Array<Ray> &rays)
{
  DRAY_LOG_OPEN("surface_intersection");

  Array<RayHit> hits = intersect_mesh_faces(rays, mesh);

  DRAY_LOG_CLOSE();
  return hits;
}

}};//naemespace dray::ray_tracing