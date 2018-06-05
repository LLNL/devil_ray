#include <dray/mfem_mesh.hpp>
#include <dray/policies.hpp>
#include <dray/point_location.hpp>
#include <dray/vec.hpp>
#include <dray/utils/data_logger.hpp>
#include <dray/utils/timer.hpp>

namespace dray
{

namespace detail
{

bool is_positive_basis(const mfem::FiniteElementCollection* fec)
{
  // HACK: Check against several common expected FE types

  if(fec == nullptr)
  {
    return false;
  }

  if(const mfem::H1_FECollection* h1Fec =
       dynamic_cast<const mfem::H1_FECollection*>(fec))
  {
    return h1Fec->GetBasisType() == mfem::BasisType::Positive;
  }

  if(const mfem::L2_FECollection* l2Fec =
       dynamic_cast<const mfem::L2_FECollection*>(fec))
  {
    return l2Fec->GetBasisType() == mfem::BasisType::Positive;
  }

  if( dynamic_cast<const mfem::NURBSFECollection*>(fec)       ||
      dynamic_cast<const mfem::LinearFECollection*>(fec)      ||
      dynamic_cast<const mfem::QuadraticPosFECollection*>(fec) )
  {
    return true;
  }

  return false;
}

/*!
 * \brief Utility function to get a positive (i.e. Bernstein)
 * collection of bases corresponding to the given FiniteElementCollection.
 *
 * \return A pointer to a newly allocated FiniteElementCollection
 * corresponding to \a fec
 * \note   It is the user's responsibility to deallocate this pointer.
 * \pre    \a fec is not already positive
 */
static mfem::FiniteElementCollection* get_pos_fec(
  const mfem::FiniteElementCollection* fec,
  int order,
  int dim,
  int map_type)
{
  //SLIC_CHECK_MSG( !isPositiveBasis( fec),
  //                "This function is only meant to be called "
  //                "on non-positive finite element collection" );

  // Attempt to find the corresponding positive H1 fec
  if(dynamic_cast<const mfem::H1_FECollection*>(fec))
  {
    return new mfem::H1_FECollection(order, dim, mfem::BasisType::Positive);
  }

  // Attempt to find the corresponding positive L2 fec
  if(dynamic_cast<const mfem::L2_FECollection*>(fec))
  {
    // should we throw a not supported error here?
    return new mfem::L2_FECollection(order, dim, mfem::BasisType::Positive,
                                     map_type);
  }

  // Attempt to find the corresponding quadratic or cubic fec
  // Note: Linear FECollections are positive
  if(dynamic_cast<const mfem::QuadraticFECollection*>(fec) ||
     dynamic_cast<const mfem::CubicFECollection*>(fec) )
  {
    //SLIC_ASSERT( order == 2 || order == 3);
    return new mfem::H1_FECollection(order, dim, mfem::BasisType::Positive);
  }

  // Give up -- return NULL
  return nullptr;
}

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
    int geom_type = mesh->GetElementBaseGeometry(0);
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
{

}

MFEMMesh::MFEMMesh(mfem::Mesh *mesh)
{
  // only support 3d for now
  assert(mesh->Dimension() == 3);

  m_mesh = mesh;
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

MFEMMesh::~MFEMMesh()
{

}
  
template<typename T>
void
MFEMMesh::intersect(Ray<T> &rays)
{

}

AABB 
MFEMMesh::get_bounds()
{
  return m_bvh.m_bounds;
}

template<typename T>
void
MFEMMesh::locate(Array<Vec<T,3>> &points)
{
  PointLocator locator(m_bvh);  
  locator.locate_candidates(points);
}

void
MFEMMesh::print_self()
{
  std::cout<<"MFEM Mesh :\n";
  if(m_is_high_order) std::cout<<" high order\n";
  else std::cout<<"  low order\n";
  std::cout<<"  Elems : "<<m_mesh->GetNE()<<"\n"; 
  std::cout<<"  Verts : "<<m_mesh->GetNV()<<"\n"; 
}

// explicit instantiations
template void MFEMMesh::intersect(ray32 &rays);
template void MFEMMesh::intersect(ray64 &rays);
template void MFEMMesh::locate(Array<Vec<float32,3>> &points);
template void MFEMMesh::locate(Array<Vec<float64,3>> &points);
} // namespace dray
