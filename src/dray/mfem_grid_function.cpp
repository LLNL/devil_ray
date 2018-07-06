#include <dray/mfem_grid_function.hpp>
#include <dray/shading_context.hpp>
#include <dray/array.hpp>
#include <dray/array_utils.hpp>
#include <dray/math.hpp>
#include <dray/policies.hpp>
#include <dray/types.hpp>
#include <dray/utils/mfem_utils.hpp>

namespace dray
{

MFEMGridFunction::MFEMGridFunction(mfem::GridFunction *gf)
  : _m_pos_nodes(nullptr)
{

  /// bool is_high_order =
  ///    (gf != nullptr) && (mesh->GetNE() > 0);
  /// if(!is_high_order) std::cout<<"NOT High Order\n";
  
  // Sanity checks
  /// assert(is_high_order);
  assert(gf != nullptr);

  /// Generate (or access existing) positive (Bernstein) nodal grid function
  const mfem::FiniteElementSpace *nodal_fe_space = gf->FESpace();
  if (nodal_fe_space == nullptr) { std::cerr << "GridFunction(): nodal_fe_space is NULL!" << std::endl; }

  m_pos_nodes = nullptr;

  const mfem::FiniteElementCollection *nodal_fe_coll = nodal_fe_space->FEColl();
  if (nodal_fe_coll == nullptr) { std::cerr << "GridFunction(): nodal_fe_coll is NULL!" << std::endl; }

  // Check if grid function is positive, if not create positive grid function
  if( detail::is_positive_basis( nodal_fe_coll ) )
  {
    std::cerr<<"Positive\n";
    m_pos_nodes = gf;
  }
  else
  {
    std::cerr<<"attemping to get positive basis\n";
    // Assume that all elements of the mesh have the same order and geom type
    mfem::Mesh *gf_mesh = nodal_fe_space->GetMesh();
    if (gf_mesh == nullptr) { std::cerr << "GridFunction(): gf_mesh is NULL!" << std::endl; }

    int order = nodal_fe_space->GetOrder(0);
    int dim = gf_mesh->Dimension();
    int geom_type = gf_mesh->GetElementBaseGeometry(0);
    int map_type =
      (nodal_fe_coll != nullptr)
      ? nodal_fe_coll->FiniteElementForGeometry(geom_type)->GetMapType()
      : static_cast<int>(mfem::FiniteElement::VALUE);

    mfem::FiniteElementCollection* pos_fe_coll =
        detail::get_pos_fec(nodal_fe_coll, 
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
      //DEBUG
      std::cerr << "so far so good... pos_fe_coll is not null. Making FESpace and GridFunction." << std::endl;
      const int dims = nodal_fe_space->GetVDim();
      // Create a positive (Bernstein) grid function for the nodes
      mfem::FiniteElementSpace* pos_fe_space =
        new mfem::FiniteElementSpace(gf_mesh, pos_fe_coll, dims);
      mfem::GridFunction *pos_nodes = new mfem::GridFunction(pos_fe_space);

      // m_pos_nodes takes ownership of pos_fe_coll's memory (and pos_fe_space's memory)
      pos_nodes->MakeOwner(pos_fe_coll);

      // Project the nodal grid function onto this
      pos_nodes->ProjectGridFunction(*gf);

      m_pos_nodes = pos_nodes;
      _m_pos_nodes.reset(pos_nodes);  // Use the std::shared_ptr so that somebody owns pos_nodes.
    }
    //DEBUG
    else std::cerr << "not good... pos_fe_coll is NULL. Could not make FESpace or GridFunction." << std::endl;

    //DEBUG
    if (!m_pos_nodes)
    {
      std::cerr << "GridFunction(): Construction failed;  m_pos_nodes is NULL!" << std::endl;
    }

  }
}

MFEMGridFunction::~MFEMGridFunction()
{
}


/*
 * Returns shading context of size rays.
 * This keeps image-bound buffers aligned with rays.
 * For inactive rays, the is_valid flag is set to false.
 */
template<typename T>
ShadingContext<T>
MFEMGridFunction::get_shading_context(Ray<T> &rays) const
{
  const int32 size_rays = rays.size();
  const int32 size_active_rays = rays.m_active_rays.size();

  ShadingContext<T> shading_ctx;
  shading_ctx.resize(size_rays);

  // Initialize outputs to well-defined dummy values (except m_pixel_id and m_ray_dir; see below).
  const Vec<T,3> one_two_three = {123., 123., 123.};
  array_memset_vec(shading_ctx.m_hit_pt, one_two_three);
  array_memset_vec(shading_ctx.m_normal, one_two_three);
  array_memset(shading_ctx.m_sample_val, static_cast<T>(-3.14));
  array_memset(shading_ctx.m_is_valid, static_cast<int32>(0));   // All are initialized to "invalid."
  
  // Adopt the fields (m_pixel_id) and (m_dir) from rays to intersection_ctx.
  shading_ctx.m_pixel_id = rays.m_pixel_id;
  shading_ctx.m_ray_dir = rays.m_dir;

  // TODO cache this in a field of MFEMGridFunction.
  T field_min, field_max;
  field_bounds(field_min, field_max);
  T field_range_rcp = rcp_safe(field_max - field_min);

  const int32 *hit_idx_ptr = rays.m_hit_idx.get_host_ptr_const();
  const Vec<T,3> *hit_ref_pt_ptr = rays.m_hit_ref_pt.get_host_ptr_const();

  int32 *is_valid_ptr = shading_ctx.m_is_valid.get_host_ptr();
  T *sample_val_ptr = shading_ctx.m_sample_val.get_host_ptr();
  Vec<T,3> *normal_ptr = shading_ctx.m_normal.get_host_ptr();
  //Vec<T,3> *hit_pt_ptr = shading_ctx.m_hit_pt.get_host_ptr();

  const int32 *active_rays_ptr = rays.m_active_rays.get_host_ptr_const();

  RAJA::forall<for_cpu_policy>(RAJA::RangeSegment(0, size_active_rays), [=] (int32 aray_idx)
  {
    const int32 ray_idx = active_rays_ptr[aray_idx];

    if (hit_idx_ptr[ray_idx] == -1)
    {
      // Sample is not in an element.
      is_valid_ptr[ray_idx] = 0;
    }
    else
    {
      // Sample is in an element.
      is_valid_ptr[ray_idx] = 1;

      const int32 elt_id = hit_idx_ptr[ray_idx];

      // Convert hit_ref_pt to double[3], then to mfem::IntegrationPoint..
      double ref_pt[3];
      ref_pt[0] = hit_ref_pt_ptr[ray_idx][0];
      ref_pt[1] = hit_ref_pt_ptr[ray_idx][1];
      ref_pt[2] = hit_ref_pt_ptr[ray_idx][2];

      mfem::IntegrationPoint ip;
      ip.Set(ref_pt, 3);

      // Get scalar field value and copy to output.
      const T field_val = m_pos_nodes->GetValue(elt_id, ip);
      sample_val_ptr[ray_idx] = (field_val - field_min) * field_range_rcp;

      // Get gradient vector of scalar field.
      mfem::FiniteElementSpace *fe_space = GetGridFunction()->FESpace();
      mfem::IsoparametricTransformation elt_trans;
      //TODO Follow up: I wish there were a const method for this.
      //I purposely used the (int, IsoparametricTransformation *) form to avoid mesh caching.
      fe_space->GetElementTransformation(elt_id, &elt_trans);
      elt_trans.SetIntPoint(&ip);
      mfem::Vector grad_vec;
      m_pos_nodes->GetGradient(elt_trans, grad_vec);
      
      // Normalize gradient vector and copy to output.
      Vec<T,3> gradient = {static_cast<T>(grad_vec[0]),
                           static_cast<T>(grad_vec[1]),
                           static_cast<T>(grad_vec[2])};
      T gradient_mag = gradient.magnitude();
      gradient.normalize();   //TODO What if the gradient is (0,0,0)?
      normal_ptr[ray_idx] = gradient;

      //TODO store the magnitude of the gradient if that is desired.

      //TODO compute hit point using ray origin, direction, and distance.
    }
  });

  return shading_ctx;
}


template<typename T>
void
MFEMGridFunction::field_bounds(T &lower, T &upper, int32 comp) const
{
  // The idea is...
  // Since we have forced the grid function to use a positive basis,
  // the global maximum and minimum are guaranteed to be found on nodes/vertices.

  ////int32 dofs = m_pos_nodes->FESpace()->GetNDofs();   //Not sure if dofs is the same as nval.Size()???
  mfem::Vector node_vals;
  m_pos_nodes->GetNodalValues(node_vals, comp);

  RAJA::ReduceMin<reduce_policy, T> comp_min(infinity32());
  RAJA::ReduceMax<reduce_policy, T> comp_max(neg_infinity32());

  Array<double> node_val_array(node_vals.GetData(), node_vals.Size());
  const double *node_val_ptr = node_val_array.get_device_ptr_const();

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, node_vals.Size()), [=] DRAY_LAMBDA (int32 ii)
  {
    comp_min.min(node_val_ptr[ii]);
    comp_max.max(node_val_ptr[ii]);
  });

  lower = comp_min.get();
  upper = comp_max.get();
}

template<typename T, int32 S>
void
MFEMGridFunction::field_bounds(Vec<T,S> &lower, Vec<T,S> &upper) const
{
  //TODO  I don't know how to do the vector field version yet.
  // Try using GetVectorNodalValues().
  // The question is: where are the different vector components?
}


template void MFEMGridFunction::field_bounds(float32 &lower, float32 &upper, int32 comp) const;
template void MFEMGridFunction::field_bounds(float64 &lower, float64 &upper, int32 comp) const;

template ShadingContext<float32> MFEMGridFunction::get_shading_context(Ray<float32> &rays) const;
template ShadingContext<float64> MFEMGridFunction::get_shading_context(Ray<float64> &rays) const;


} // namespace dray
