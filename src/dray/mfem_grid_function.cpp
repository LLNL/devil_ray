#include <dray/mfem_grid_function.hpp>
#include <dray/array.hpp>
#include <dray/policies.hpp>
#include <dray/types.hpp>
#include <dray/utils/mfem_utils.hpp>

namespace dray
{


MFEMGridFunction::MFEMGridFunction(mfem::GridFunction *gf)
{

  /// bool is_high_order =
  ///    (gf != nullptr) && (mesh->GetNE() > 0);
  /// if(!is_high_order) std::cout<<"NOT High Order\n";
  
  // Sanity checks
  /// assert(is_high_order);
  assert(gf != nullptr);

  /// Generate (or access existing) positive (Bernstein) nodal grid function
  const mfem::FiniteElementSpace *nodal_fe_space = gf->FESpace();
  if (nodal_fe_space == nullptr) { std::cout << "GridFunction(): nodal_fe_space is NULL!" << std::endl; }

  m_pos_nodes = nullptr;
  m_delete_nodes = false;

  const mfem::FiniteElementCollection *nodal_fe_coll = nodal_fe_space->FEColl();
  if (nodal_fe_coll == nullptr) { std::cout << "GridFunction(): nodal_fe_coll is NULL!" << std::endl; }

  // Check if grid function is positive, if not create positive grid function
  if( detail::is_positive_basis( nodal_fe_coll ) )
  {
    std::cout<<"Positive\n";
    m_pos_nodes = gf;
  }
  else
  {
    std::cout<<"attemping to get positive basis\n";
    // Assume that all elements of the mesh have the same order and geom type
    mfem::Mesh *gf_mesh = nodal_fe_space->GetMesh();
    if (gf_mesh == nullptr) { std::cout << "GridFunction(): gf_mesh is NULL!" << std::endl; }

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
      const int dims = nodal_fe_space->GetVDim();
      // Create a positive (Bernstein) grid function for the nodes
      mfem::FiniteElementSpace* pos_fe_space =
        new mfem::FiniteElementSpace(gf_mesh, pos_fe_coll, dims);
      mfem::GridFunction *pos_nodes = new mfem::GridFunction(pos_fe_space);

      // m_pos_nodes takes ownership of pos_fe_coll's memory (and pos_fe_space's memory)
      pos_nodes->MakeOwner(pos_fe_coll);
      m_delete_nodes = true;

      // Project the nodal grid function onto this
      pos_nodes->ProjectGridFunction(*gf);

      m_pos_nodes = pos_nodes;
    }

    //DEBUG
    if (!m_pos_nodes)
    {
      std::cout << "GridFunction(): Construction failed;  m_pos_nodes is NULL!" << std::endl;
    }

  }
}


template<typename T>
void
MFEMGridFunction::get_bounds(T &lower, T &upper, int32 comp)
{
  // The idea is...
  // Since we have forced the grid function to use a positive basis,
  // the global maximum and minimum are guaranteed to be found on nodes/vertices.

  ////int32 dofs = m_pos_nodes->FESpace()->GetNDofs();   //Not sure if dofs is the same as nval.Size()???
  mfem::Vector node_vals;
  m_pos_nodes->GetNodalValues(node_vals, comp);

  RAJA::ReduceMin<reduce_cpu_policy, T> comp_min(infinity32());
  RAJA::ReduceMax<reduce_cpu_policy, T> comp_max(neg_infinity32());

  Array<double> node_val_array(node_vals.GetData(), node_vals.Size());
  const double *node_val_ptr = node_val_array.get_device_ptr_const();

  RAJA::forall<for_cpu_policy>(RAJA::RangeSegment(0, node_vals.Size()), [=] (int32 ii)
  {
    comp_min.min(node_val_ptr[ii]);
    comp_max.max(node_val_ptr[ii]);
  });

  lower = comp_min.get();
  upper = comp_max.get();
}

template<typename T, int32 S>
void
MFEMGridFunction::get_bounds(Vec<T,S> &lower, Vec<T,S> &upper)
{
  //TODO  I don't know how to do the vector field version yet.
  // Try using GetVectorNodalValues().
  // The question is: where are the different vector components?
}



} // namespace dray
