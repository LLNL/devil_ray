#include <dray/mfem_mesh.hpp>
#include <dray/policies.hpp>
#include <dray/error.hpp>
#include <dray/point_location.hpp>
#include <dray/vec.hpp>
#include <dray/array_utils.hpp>
#include <dray/utils/mfem_utils.hpp>
#include <dray/utils/data_logger.hpp>
#include <dray/utils/timer.hpp>

namespace dray
{

namespace detail
{

/*!
 * Helper function to initialize the bounding boxes for a low-order mfem mesh
 *
 * \param bboxScaleFactor Scale factor for expanding bounding boxes
 * \note This function can only be called before the class is initialized
 * \pre This function can only be called when m_isHighOrder is false
 * \pre bboxScaleFactor must be greater than or equal to 1.
 *
 * \sa computeBoundingBoxes()
 */
void compute_low_order_AABBs( mfem::Mesh *mesh,
                              double bbox_scale,
                              Array< AABB >& aabbs)
{
  //SLIC_ASSERT( !m_isHighOrder );
  //SLIC_ASSERT( bboxScaleFactor >= 1. );

  /// For each element, compute bounding box, and overall mesh bbox
  const int num_els = mesh->GetNE();

  aabbs.resize(num_els);
  AABB *aabb_ptr = aabbs.get_host_ptr();

  // This is on the CPU only. Note: do not use DRAY_LAMBDA
  RAJA::forall<for_cpu_policy>(RAJA::RangeSegment(0, num_els), [=] (int32 elem)
  {
    AABB bbox;

    mfem::Element* elt = mesh->GetElement(elem);
    int* elt_verts = elt->GetVertices();
    for(int32 i = 0 ; i< elt->GetNVertices() ; ++i)
    {
      int vIdx = elt_verts[i];
      double *vertex = mesh->GetVertex( vIdx );
      Vec3f vec3f;
      vec3f[0] = vertex[0];
      vec3f[1] = vertex[1];
      vec3f[2] = vertex[2];
      bbox.include( vec3f );
    }

    // scale the bounding box to account for numerical noise
    bbox.scale(bbox_scale);
    aabb_ptr[elem] = bbox;
  });
}

void compute_high_order_AABBs( mfem::Mesh *mesh,
                               double bbox_scale,
                               Array< AABB >& aabbs)
{
  DRAY_LOG_OPEN("compute_high_order_aabbs");
  Timer tot_time;

  bool is_high_order =
     (mesh->GetNodalFESpace() != nullptr) && (mesh->GetNE() > 0);
  if(!is_high_order) std::cout<<"NOT HO\n";

  // Sanity checks
  assert(is_high_order);

  /// Generate (or access existing) positive (Bernstein) nodal grid function
  const mfem::FiniteElementSpace* nodal_fe_space = mesh->GetNodalFESpace();

  mfem::GridFunction* pos_nodes = nullptr;
  bool delete_nodes = false;

  const mfem::FiniteElementCollection* nodal_fe_coll = nodal_fe_space->FEColl();
  std::cout<<"Boom\n";
  // Check if grid function is positive, if not create positive grid function
  if( is_positive_basis( nodal_fe_coll ) )
  {
    std::cout<<"Positive\n";
    pos_nodes = mesh->GetNodes();
  }
  else
  {
    std::cout<<"attemping to get positive basis\n";
    // Assume that all elements of the mesh have the same order and geom type
    int order = nodal_fe_space->GetOrder(0);
    int dim = mesh->Dimension();
    mfem::Geometry::Type geom_type = mesh->GetElementBaseGeometry(0);
    int map_type =
      (nodal_fe_coll != nullptr)
      ? nodal_fe_coll->FiniteElementForGeometry(geom_type)->GetMapType()
      : static_cast<int>(mfem::FiniteElement::VALUE);

    mfem::FiniteElementCollection* pos_fe_coll =
      get_pos_fec(nodal_fe_coll,
                  order,
                  dim,
                  map_type);

    //SLIC_ASSERT_MSG(
    //  pos_fe_coll != AXOM_NULLPTR,
    //  "Problem generating a positive finite element collection "
    //  << "corresponding to the mesh's '"<< nodal_fe_coll->Name()
    //  << "' finite element collection.");

    if(pos_fe_coll != nullptr)
    {
      const int dims = 3;
      // Create a positive (Bernstein) grid function for the nodes
      mfem::FiniteElementSpace* pos_fe_space =
        new mfem::FiniteElementSpace(mesh, pos_fe_coll, dims);
      pos_nodes = new mfem::GridFunction(pos_fe_space);

      // m_bernsteinNodes takes ownership of pos_fe_coll's memory
      pos_nodes->MakeOwner(pos_fe_coll);
      delete_nodes = true;

      // Project the nodal grid function onto this
      pos_nodes->ProjectGridFunction(*(mesh->GetNodes()));
    }

  }

  // Output some information
  //SLIC_INFO(
  //  "Mesh nodes fec -- "
  //  << nodal_fe_coll->Name()
  //  << " with ordering " << nodal_fe_space->GetOrdering()
  //  << "\n\t -- Positive nodes are fec -- "
  //  << pos_nodes->FESpace()->FEColl()->Name()
  //  << " with ordering " << pos_nodes->FESpace()->GetOrdering() );
  std::cout<<
    "Mesh nodes fec -- "
    << nodal_fe_coll->Name()
    << " with ordering " << nodal_fe_space->GetOrdering()
    << "\n\t -- Positive nodes are fec -- "
    << pos_nodes->FESpace()->FEColl()->Name()
    << " with ordering " << pos_nodes->FESpace()->GetOrdering()<<"\n";


  /// For each element, compute bounding box, and overall mesh bbox
  mfem::FiniteElementSpace* fes = pos_nodes->FESpace();
  const int num_els = mesh->GetNE();

  aabbs.resize(num_els);
  AABB *aabb_ptr = aabbs.get_host_ptr();

  // This is on the CPU only. Note: do not use DRAY_LAMBDA
  RAJA::forall<for_cpu_policy>(RAJA::RangeSegment(0, num_els), [=] (int32 elem)
  {
    AABB bbox;

    // Add each dof of the element to the bbox
    // Note: positivity of Bernstein bases ensures that convex
    //       hull of element nodes contain entire element
    mfem::Array<int> dof_indices;
    fes->GetElementDofs(elem, dof_indices);
    for(int i = 0 ; i< dof_indices.Size() ; ++i)
    {
      int nIdx = dof_indices[i];

      Vec3f pt;
      for(int j=0 ; j< 3; ++j)
        pt[j] = (*pos_nodes)(fes->DofToVDof(nIdx,j));

      bbox.include( pt );
    }

    // Slightly scale the bbox to account for numerical noise
    bbox.scale(bbox_scale);

    aabb_ptr[elem] = bbox;
  });

  /// Clean up -- deallocate grid function if necessary
  if(delete_nodes)
  {
    delete pos_nodes;
    pos_nodes = nullptr;
  }
  DRAY_LOG_ENTRY("tot_time", tot_time.elapsed());
  DRAY_LOG_CLOSE();
}
} // namespace detail

MFEMMesh::MFEMMesh()
  : m_mesh(NULL)
{

}

MFEMMesh::MFEMMesh(mfem::Mesh *mesh)
{
  this->set_mesh(mesh);
}

MFEMMesh::~MFEMMesh()
{

}

void
MFEMMesh::set_mesh(mfem::Mesh *mesh)
{
  // only support 3d for now
  assert(mesh->Dimension() == 3);

  m_mesh = mesh;
  if(m_mesh->NURBSext)
  {
    m_mesh->SetCurvature(2);
  }

  m_is_high_order =
     (mesh->GetNodalFESpace() != nullptr) && (mesh->GetNE() > 0);

  double bbox_scale = 1.000001;

  Array<AABB> aabbs;

  //AABB tot_bounds;
  std::cout<<"here\n";
  if(m_is_high_order)
  {
    std::cout<<"Higher order\n";
    detail::compute_high_order_AABBs( m_mesh,
                                      bbox_scale,
                                      aabbs);
  }
  else
  {

    std::cout<<"linear order\n";
    detail::compute_low_order_AABBs( m_mesh,
                                     bbox_scale,
                                     aabbs);
  }

  LinearBVHBuilder builder;
  m_bvh = builder.construct(aabbs);
  std::cout<<"MFEM Bounds "<<m_bvh.m_bounds<<"\n";

}


template<typename T>
void
MFEMMesh::intersect(Ray<T> &rays)
{
  if(m_mesh == nullptr)
  {
    throw DRayError("Mesh intersect: mesh cannot be null. Call set_mesh before locate");
  }
}

AABB
MFEMMesh::get_bounds()
{
  return m_bvh.m_bounds;
}

template<typename T>
void
MFEMMesh::locate(const Array<Vec<T,3>> points,
                 Array<Ray<T>> &rays)
{
  const Array<int32> active_idx = array_counting(points.size(), 0,1);
    // Assume that elt_ids and ref_pts are sized to same length as points.
  locate(points, active_idx, rays);
}

template<typename T>
void
MFEMMesh::locate(const Array<Vec<T,3>> points,
                 const Array<int32> active_idx,
                 Array<Ray<T>> &rays)
{
  if(m_mesh == nullptr)
  {
    throw DRayError("Mesh locate: mesh cannot be null. Call set_mesh before locate");
  }

  DRAY_LOG_OPEN("locate_point");

  const int size = points.size();
  const int active_size = active_idx.size();

  PointLocator locator(m_bvh);
  constexpr int32 max_candidates = 10;
  Array<int32> candidates = locator.locate_candidates(points, active_idx, max_candidates);  //Size active_size * max_candidates.

  Timer timer;

  const int *candidates_ptr = candidates.get_host_ptr_const();

  const Vec<T,3> *points_ptr = points.get_host_ptr_const();
  Ray<T> *ray_ptr = rays.get_host_ptr();

  // Initialize outputs to well-defined dummy values.
  const Vec<T,3> three_point_one_four = {3.14, 3.14, 3.14};

  const int32 *active_idx_ptr = active_idx.get_host_ptr_const();

  DRAY_LOG_ENTRY("setup", timer.elapsed());
  timer.reset();

  RAJA::forall<for_cpu_policy>(RAJA::RangeSegment(0, active_size), [=] (int32 aii)
  {
    const int32 ii = active_idx_ptr[aii];

    Ray<T> &ray = ray_ptr[ii];
    ray.m_hit_idx = -1;
    ray.m_hit_ref_pt = three_point_one_four;

    // - Use aii to index into candidates.
    // - Use ii to index into points, elt_ids, and ref_pts.

    int32 count = 0;
    int32 el_idx = candidates_ptr[aii * max_candidates + count];
    float64 pt[3];
    float64 isopar[3];
    Vec<T,3> p = points_ptr[ii];
    pt[0] = static_cast<float64>(p[0]);
    pt[1] = static_cast<float64>(p[1]);
    pt[2] = static_cast<float64>(p[2]);

    bool found_inside = false;

		int cand = 0;
    while(!found_inside && count < max_candidates && el_idx != -1)
    {
			cand++;
      // we only support 3d meshes
      constexpr int dim = 3;
      mfem::IsoparametricTransformation tr;
      m_mesh->GetElementTransformation(el_idx, &tr);
      mfem::Vector ptSpace(const_cast<double*>(pt), dim);

      mfem::IntegrationPoint ipRef;

      // Set up the inverse element transformation
      typedef mfem::InverseElementTransformation InvTransform;
      InvTransform invTrans(&tr);

      invTrans.SetSolverType( InvTransform::Newton );
      //invTrans.SetInitialGuessType(InvTransform::ClosestPhysNode);
      // TODO: this above is better but cannot be called in parallel
      invTrans.SetInitialGuessType(InvTransform::Center);

      // Status codes: {0 -> successful; 1 -> outside elt; 2-> did not converge}
      int err = invTrans.Transform(ptSpace, ipRef);

      ipRef.Get(isopar, dim);

      if (err == 0)
      {
        // Found the element. Stop search, preserving count and el_idx.
        found_inside = true;
        break;
      }
      else
      {
        // Continue searching with the next candidate.
        count++;
        el_idx = candidates_ptr[aii*max_candidates + count];
           //NOTE: This may read past end of array, but only if count >= max_candidates.
      }
    }

    // After testing each candidate, now record the result.
    if (found_inside)
    {
      ray.m_hit_idx = el_idx;
      ray.m_hit_ref_pt[0] = isopar[0];
      ray.m_hit_ref_pt[1] = isopar[1];
      ray.m_hit_ref_pt[2] = isopar[2];
    }
    else
    {
      ray.m_active = 0;
    }

		//if(cand != 0) std::cout<<"candidates "<<cand<<"\n";
  });

  DRAY_LOG_ENTRY("kernel", timer.elapsed());
  timer.reset();

  DRAY_LOG_CLOSE();
}

void
MFEMMesh::print_self()
{
  if(m_mesh == nullptr)
  {
    std::cout<<"Mesh is nullptr: call set_mesh\n";
  }
  else
  {
    std::cout<<"MFEM Mesh :\n";
    if(m_is_high_order) std::cout<<" high order\n";
    else std::cout<<"  low order\n";
    std::cout<<"  Elems : "<<m_mesh->GetNE()<<"\n";
    std::cout<<"  Verts : "<<m_mesh->GetNV()<<"\n";
    std::cout<<"  p_msh : "<<m_mesh<<"\n";
  }
}


//template<typename T>
//void
//MFEMMeshField::cast_to_isosurface(Ray<T> &rays, T isovalue, int32 guesses_per_elt)
//{
//  const int32 size_rays = rays.size();
//  const int32 size_active = rays.m_active_rays.size();
//
//  constexpr int32 max_candidates = 5;
//  const Array<int32> candidates;
//  //const Array<int32> candidates = intersect_rays(m_bvh, rays, max_candidates);   //TODO method
//
//  const int32 *candidates_ptr = candidates.get_device_ptr_const();
//  const int32 *active_rays_ptr = rays.m_active_rays.get_device_ptr_const();
//
//  RAJA::forall<for_cpu_policy>(RAJA::RangeSegment(0, size_active), [=] (int32 aii)
//  {
//    const int32 ray_idx = active_rays_ptr[aii];
//    // - Use aii to index into candidates.
//    // - Use ray_idx to index into rays.
//
//    int32 count = 0;
//    int32 el_idx = candidates_ptr[aii*max_candidates + count];
//
//    // Loop over candidate elements.
//    while (count < max_candidates && el_idx != -1)
//    {
//      // Do guesses_per_elt Newton solves per candidate.
//
//
//      int32 el_idx = candidates_ptr[aii*max_candidates + count];
//    }
//
//  });
//}

// explicit instantiations
template void MFEMMesh::intersect(ray32 &rays);
template void MFEMMesh::intersect(ray64 &rays);
template void MFEMMesh::locate(const Array<Vec<float32,3>> points, Array<Ray<float32>> &rays);
template void MFEMMesh::locate(const Array<Vec<float64,3>> points, Array<Ray<float64>> &rays);
} // namespace dray
