#include <dray/mfem2dray.hpp>
#include <dray/GridFunction/mesh.hpp>
#include <dray/GridFunction/field.hpp>
#include <dray/GridFunction/grid_function_data.hpp>
#include <dray/utils/mfem_utils.hpp>
#include <dray/policies.hpp>
#include <dray/types.hpp>

#include <iostream>

namespace dray
{

namespace detail
{
// Lexicographic ordering in MFEM is X-inner, Z-outer. I used to flip to X-outer, Z-inner.
template <int32 S>
int32 reverse_lex(int32 in_idx, int32 l)
{
  int32 out_idx = 0;
  for (int32 dim = 0; dim < S; dim++)
  {
    int32 comp = in_idx % l;
    in_idx /= l;

    out_idx *= l;
    out_idx += comp;
  }
  return out_idx;
}
} // namespace detail


template <typename T>
Mesh<T,3> import_mesh(const mfem::Mesh &mfem_mesh)
{
  int32 poly_order;
  GridFunctionData<T,3> dof_data = import_mesh<T>(mfem_mesh, poly_order);
  return Mesh<T>(dof_data, poly_order);
}

template <typename T, int32 PhysDim>
Field<T,3,PhysDim> import_field(const mfem::GridFunction &mfem_gf)
{
  int32 poly_order;
  GridFunctionData<T,PhysDim> dof_data = import_grid_function<T,PhysDim>(mfem_gf, poly_order);
  return Field<T,3,PhysDim>(dof_data, poly_order);
}

template <typename T>
Field<T,3,1> import_vector_field_component(const mfem::GridFunction &mfem_gf, int32 comp)
{
  int32 poly_order;
  GridFunctionData<T,1> dof_data = import_vector_field_component<T>(mfem_gf, comp, poly_order);
  return Field<T,3,1>(dof_data, poly_order);
}


template <typename T>
GridFunctionData<T,3> import_mesh(const mfem::Mesh &mfem_mesh, int32 &space_P)
{

  const mfem::GridFunction *mesh_nodes;
  if(mfem_mesh.Conforming())
  {
    std::cout<<"Conforming mesh\n";
  }
  else
  {
    std::cout<<"Non conforming\n";
  }
  if ((mesh_nodes = mfem_mesh.GetNodes()) != NULL)
  {
    //std::cerr << "mfem2dray import_mesh() - GetNodes() is NOT null." << std::endl;
    return import_grid_function<T,3>(*mesh_nodes, space_P);
  }
  else
  {
    //std::cerr << "mfem2dray import_mesh() - GetNodes() is NULL." << std::endl;
    space_P = 1;
    return import_linear_mesh<T>(mfem_mesh);
  }
}

template <typename T>
GridFunctionData<T,3> import_linear_mesh(const mfem::Mesh &mfem_mesh)
{
  GridFunctionData<T,3> dataset;
  //TODO resize, import, etc.
  return dataset;
}

template <typename T, int32 PhysDim>
GridFunctionData<T,PhysDim> import_grid_function(const mfem::GridFunction &_mfem_gf, int32 &space_P)
{
  bool is_gf_new;
  mfem::GridFunction *pos_gf = project_to_pos_basis(&_mfem_gf, is_gf_new);
  const mfem::GridFunction & mfem_gf = (is_gf_new ? *pos_gf : _mfem_gf);

  constexpr int32 phys_dim = PhysDim;
  GridFunctionData<T,phys_dim> dataset;

  // Access to degree of freedom mapping.
  const mfem::FiniteElementSpace *fespace = mfem_gf.FESpace();
  //printf("fespace == %x\n", fespace);

  // Access to control point data.
  const mfem::Vector &ctrl_vals = mfem_gf;
  //mfem_gf.GetTrueDofs(ctrl_vals);   // Sets size and initializes data. Might be reference.

  const int32 P = fespace->GetOrder(0);

  mfem::Array<int> zeroth_dof_set;
  fespace->GetElementDofs(0, zeroth_dof_set);

  const int32 dofs_per_element = zeroth_dof_set.Size();
  const int32 num_elements = fespace->GetNE();
  const int32 num_ctrls = ctrl_vals.Size() / phys_dim;

  // Enforce: All elements must have same number of dofs.

  mfem::Table el_dof_table( fespace->GetElementToDofTable() );
  el_dof_table.Finalize();
  const int32 all_el_dofs = el_dof_table.Size_of_connections();

  //std::cout << "all_el_dofs == " << all_el_dofs << std::endl;
  assert(all_el_dofs == num_elements * dofs_per_element);   // This is what I meant.

  // Former attempt at the above assertion.
  const int32 mfem_num_dofs = fespace->GetNDofs();

  //std::cout<<"Mfem dofs "<<mfem_num_dofs<<" "<<num_elements * dofs_per_element<<"\n";
  //std::cout<<"el 0 dof "<<dofs_per_element<<" nels "<<num_elements<<"\n";
  //std::cout<<"num_ctrls "<<num_ctrls<<"\n";
  // I could be way off base here, but dofs could be shared between elements, so the number
  // is lower than expected.
  ////assert(mfem_num_dofs == num_elements * dofs_per_element);  // You're right, these should not be equal in general.

  dataset.resize(num_elements, dofs_per_element, num_ctrls);

  // Are these MFEM data structures thread-safe?  TODO

  int32 stride_pdim;
  int32 stride_ctrl;
  if (fespace->GetOrdering() == mfem::Ordering::byNODES)  // XXXX YYYY ZZZZ
  {
    //printf("Calculating stride byNODES\n");
    //stride_pdim = num_elements;
    stride_pdim = num_ctrls;
    stride_ctrl = 1;
  }
  else                                                    // XYZ XYZ XYZ XYZ
  {
    //printf("Calculating stride byVDIM\n");
    stride_pdim = 1;
    stride_ctrl = phys_dim;
  }

  //
  // Import degree of freedom values.
  //
  Vec<T,phys_dim> *ctrl_val_ptr = dataset.m_values.get_host_ptr();
  ///RAJA::forall<for_cpu_policy>(RAJA::RangeSegment(0, num_ctrls), [=] (int32 ctrl_id)
  for (int32 ctrl_id = 0; ctrl_id < num_ctrls; ctrl_id++)
  {
    // TODO get internal representation of the mfem memory, so we can access in a device function.
    //
    for (int32 pdim = 0; pdim < phys_dim; pdim++)
    {
      ctrl_val_ptr[ctrl_id][pdim] = ctrl_vals( pdim * stride_pdim + ctrl_id * stride_ctrl );
    }
  }
  ///});

  // DRAY and MFEM may store degrees of freedom in different orderings.
  const bool use_dof_map = fespace->Conforming();
  mfem::H1Pos_HexahedronElement fe_prototype(P);
  const mfem::Array<int> &fe_dof_map = fe_prototype.GetDofMap();

  // DEBUG
  //printf("use_dof_map == %d\n", use_dof_map);
  //printf("fe_dof_map:   ");
  //for (int32 map_idx = 0; map_idx < fe_dof_map.Size(); map_idx++)
  //{
  //  printf("%d ", fe_dof_map[map_idx]);
  //}
  //printf("\n");


  //// //DEBUG
  //// std::cout << "Element values." << std::endl;

  //
  // Import degree of freedom mappings.
  //
  //std::cout<<"NUM ELEMENTS "<<num_elements<<"\n";
  //if(use_dof_map)
  //{
  //  std::cout<<"USING DOF map\n";
  //}
  //else
  //{
  //   std::cout<<"NOT USING DOF map\n";
  //}
  int32 *ctrl_idx_ptr = dataset.m_ctrl_idx.get_host_ptr();
  ///RAJA::forall<for_cpu_policy>(RAJA::RangeSegment(0, num_elements), [=] (int32 el_id)
  for (int32 el_id = 0; el_id < num_elements; el_id++)
  {
    // TODO get internal representation of the mfem memory, so we can access in a device function.
    //
    mfem::Array<int> el_dof_set;
    fespace->GetElementDofs(el_id, el_dof_set);
    int dof_size = el_dof_set.Size();
    //std::cout<<"DOF Set "<<dof_size<<"\n";
    for (int32 dof_id = el_id * dofs_per_element, el_dof_id = 0;
         el_dof_id < dofs_per_element;
         dof_id++, el_dof_id++)
    {
        // Maintain same lexicographic order as MFEM (X-inner:Z-outer).
      const int32 el_dof_id_lex = el_dof_id;
        // Maybe there's a better practice than this inner conditional.
      const int32 mfem_el_dof_id = use_dof_map ? fe_dof_map[el_dof_id_lex] : el_dof_id_lex;
      //if(mfem_el_dof_id >= dof_size) std::cout<<"BAD INDEX "<<mfem_el_dof_id<<"\n";
      ctrl_idx_ptr[dof_id] = el_dof_set[mfem_el_dof_id];

      //// //DEBUG
      //// std::cout << ctrl_val_ptr[ ctrl_idx_ptr[dof_id] ] << "	";
    }
    /////std::cout << std::endl << std::endl;
  }
  ///});


  if (is_gf_new)
  {
    delete pos_gf;
  }

  space_P = P;
  return dataset;
}


//
// import_vector_field_component()
//
template <typename T>
GridFunctionData<T,1> import_vector_field_component(const mfem::GridFunction &_mfem_gf, int32 comp, int32 &space_P)
{
  bool is_gf_new;
  mfem::GridFunction *pos_gf = project_to_pos_basis(&_mfem_gf, is_gf_new);
  const mfem::GridFunction & mfem_gf = (is_gf_new ? *pos_gf : _mfem_gf);

  GridFunctionData<T,1> dataset;

  const int32 vec_dim = mfem_gf.VectorDim();

  // Access to degree of freedom mapping.
  const mfem::FiniteElementSpace *fespace = mfem_gf.FESpace();

  // Access to control point data.
  const mfem::Vector &ctrl_vals = mfem_gf;
  //mfem_gf.GetTrueDofs(ctrl_vals);   // Sets size and initializes data. Might be reference.

  //DEBUG
  //printf("ctrl_vals.Size() == %d,  mfem_gf.VectorDim() == %d\n",
  //    ctrl_vals.Size(), mfem_gf.VectorDim());

  const int32 P = fespace->GetOrder(0);

  mfem::Array<int> zeroth_dof_set;
  fespace->GetElementDofs(0, zeroth_dof_set);

  const int32 dofs_per_element = zeroth_dof_set.Size();
  const int32 num_elements = fespace->GetNE();
  const int32 num_ctrls = ctrl_vals.Size() / vec_dim;

  // Enforce: All elements must have same number of dofs.

  mfem::Table el_dof_table( fespace->GetElementToDofTable() );
  el_dof_table.Finalize();
  const int32 all_el_dofs = el_dof_table.Size_of_connections();

  //std::cout << "all_el_dofs == " << all_el_dofs << std::endl;
  assert(all_el_dofs == num_elements * dofs_per_element);   // This is what I meant.

  /// // Former attempt at the above assertion.
  /// const int32 mfem_num_dofs = fespace->GetNDofs();

  /// std::cout<<"Mfem dofs "<<mfem_num_dofs<<" "<<num_elements * dofs_per_element<<"\n";
  /// std::cout<<"el 0 dof "<<dofs_per_element<<" nels "<<num_elements<<"\n";
  /// std::cout<<"num_ctrls "<<num_ctrls<<"\n";
  /// // I could be way off base here, but dofs could be shared between elements, so the number
  /// // is lower than expected.
  /// ////assert(mfem_num_dofs == num_elements * dofs_per_element);  // You're right, these should not be equal in general.

  dataset.resize(num_elements, dofs_per_element, num_ctrls);

  // Are these MFEM data structures thread-safe?  TODO

  int32 stride_pdim;
  int32 stride_ctrl;
  if (fespace->GetOrdering() == mfem::Ordering::byNODES)  // XXXX YYYY ZZZZ
  {
    //printf("Calculating stride byNODES\n");
    stride_pdim = num_ctrls;
    stride_ctrl = 1;
  }
  else                                                    // XYZ XYZ XYZ XYZ
  {
    //printf("Calculating stride byVDIM\n");
    stride_pdim = 1;
    stride_ctrl = vec_dim;
  }

   //
   // Import degree of freedom values.
   //
   Vec<T,1> *ctrl_val_ptr = dataset.m_values.get_host_ptr();
   ///RAJA::forall<for_cpu_policy>(RAJA::RangeSegment(0, num_ctrls), [=] (int32 ctrl_id)
   for (int32 ctrl_id = 0; ctrl_id < num_ctrls; ctrl_id++)
   {
     // TODO get internal representation of the mfem memory, so we can access in a device function.
     //
     ctrl_val_ptr[ctrl_id][0] = ctrl_vals( comp * stride_pdim + ctrl_id * stride_ctrl );
   }
   ///});

   // DRAY and MFEM may store degrees of freedom in different orderings.
   const bool use_dof_map = fespace->Conforming();
   mfem::H1Pos_HexahedronElement fe_prototype(P);
   const mfem::Array<int> &fe_dof_map = fe_prototype.GetDofMap();

   // DEBUG
   //printf("use_dof_map == %d\n", use_dof_map);
   //printf("fe_dof_map:   ");
   //for (int32 map_idx = 0; map_idx < fe_dof_map.Size(); map_idx++)
   //{
   //  printf("%d ", fe_dof_map[map_idx]);
   //}
   //printf("\n");


   //// //DEBUG
   //// std::cout << "Element values." << std::endl;

   //
   // Import degree of freedom mappings.
   //
   int32 *ctrl_idx_ptr = dataset.m_ctrl_idx.get_host_ptr();
   ///RAJA::forall<for_cpu_policy>(RAJA::RangeSegment(0, num_elements), [=] (int32 el_id)
   for (int32 el_id = 0; el_id < num_elements; el_id++)
   {
     // TODO get internal representation of the mfem memory, so we can access in a device function.
     //
     mfem::Array<int> el_dof_set;
     fespace->GetElementDofs(el_id, el_dof_set);
     for (int32 dof_id = el_id * dofs_per_element, el_dof_id = 0;
          el_dof_id < dofs_per_element;
          dof_id++, el_dof_id++)
     {
         // Maintain same lexicographic order as MFEM (X-inner:Z-outer).
       const int32 el_dof_id_lex = el_dof_id;
         // Maybe there's a better practice than this inner conditional.
       const int32 mfem_el_dof_id = use_dof_map ? fe_dof_map[el_dof_id_lex] : el_dof_id_lex;
       ctrl_idx_ptr[dof_id] = el_dof_set[mfem_el_dof_id];

       //// //DEBUG
       //// std::cout << ctrl_val_ptr[ ctrl_idx_ptr[dof_id] ] << "	";
     }
     /////std::cout << std::endl << std::endl;
   }
   ///});


   if (is_gf_new)
   {
     delete pos_gf;
   }

   space_P = P;
   return dataset;
 }




template <typename T>
GridFunctionData<T,1> import_grid_function_field(const mfem::GridFunction &mfem_gf)
{
  GridFunctionData<T,1> dataset;
  //TODO resize, import, etc.
  return dataset;
}


// Explicit instantiations
template GridFunctionData<float32,3>
import_mesh<float32>(const mfem::Mesh &mfem_mesh, int32 &space_P);

template GridFunctionData<float32,3>
import_linear_mesh<float32>(const mfem::Mesh &mfem_mesh);

template GridFunctionData<float32,1>
import_grid_function<float32,1>(const mfem::GridFunction &mfem_gf, int32 &field_P);

template GridFunctionData<float32,3>
import_grid_function<float32,3>(const mfem::GridFunction &mfem_gf, int32 &field_P);

template GridFunctionData<float32,1>
import_vector_field_component<float32>(const mfem::GridFunction &mfem_gf, int32 comp, int32 &field_P);

template Mesh<float32,3>
import_mesh<float32>(const mfem::Mesh &mfem_mesh);

template Field<float32,3,1>
import_field<float32,1>(const mfem::GridFunction &mfem_gf);

template Field<float32,3,3>
import_field<float32,3>(const mfem::GridFunction &mfem_gf);

template Field<float32,3,1>
import_vector_field_component<float32>(const mfem::GridFunction &mfem_gf, int32 comp);



template GridFunctionData<float64,3>
import_mesh<float64>(const mfem::Mesh &mfem_mesh, int32 &space_P);

template GridFunctionData<float64,3> import_linear_mesh<float64>(const mfem::Mesh &mfem_mesh);
template GridFunctionData<float64,1> import_grid_function<float64,1>(const mfem::GridFunction &mfem_gf, int32 &field_P);
template GridFunctionData<float64,3> import_grid_function<float64,3>(const mfem::GridFunction &mfem_gf, int32 &field_P);
template GridFunctionData<float64,1> import_vector_field_component<float64>(const mfem::GridFunction &mfem_gf, int32 comp, int32 &field_P);

template Mesh<float64,3> import_mesh<float64>(const mfem::Mesh &mfem_mesh);
template Field<float64,3,1> import_field<float64,1>(const mfem::GridFunction &mfem_gf);
template Field<float64,3,3> import_field<float64,3>(const mfem::GridFunction &mfem_gf);
template Field<float64,3,1> import_vector_field_component<float64>(const mfem::GridFunction &mfem_gf, int32 comp);




//
// project_to_pos_basis()
//
  // If is_new was set to true, the caller is responsible for deleting the returned pointer.
  // If is_new was set to false, then the returned value is null, and the caller should use gf.
mfem::GridFunction * project_to_pos_basis(const mfem::GridFunction *gf, bool &is_new)
{
  mfem::GridFunction * out_pos_gf = nullptr;
  is_new = false;

  /// bool is_high_order =
  ///    (gf != nullptr) && (mesh->GetNE() > 0);
  /// if(!is_high_order) std::cout<<"NOT High Order\n";

  // Sanity checks
  /// assert(is_high_order);
  assert(gf != nullptr);

  /// Generate (or access existing) positive (Bernstein) nodal grid function
  const mfem::FiniteElementSpace *nodal_fe_space = gf->FESpace();
  if (nodal_fe_space == nullptr) { std::cerr << "project_to_pos_basis(): nodal_fe_space is NULL!" << std::endl; }

  const mfem::FiniteElementCollection *nodal_fe_coll = nodal_fe_space->FEColl();
  if (nodal_fe_coll == nullptr) { std::cerr << "project_to_pos_basis(): nodal_fe_coll is NULL!" << std::endl; }

  // Check if grid function is positive, if not create positive grid function
  if( detail::is_positive_basis( nodal_fe_coll ) )
  {
    //std::cerr<<"Already positive.\n";
    is_new = false;
    out_pos_gf = nullptr;
  }
  else
  {
    //std::cerr<<"Attemping to convert to positive basis.\n";
    // Assume that all elements of the mesh have the same order and geom type
    mfem::Mesh *gf_mesh = nodal_fe_space->GetMesh();
    if (gf_mesh == nullptr) { std::cerr << "project_to_pos_basis(): gf_mesh is NULL!" << std::endl; }

    int order = nodal_fe_space->GetOrder(0);
    int dim = gf_mesh->Dimension();
    mfem::Geometry::Type geom_type = gf_mesh->GetElementBaseGeometry(0);
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
      //std::cerr << "Good so far... pos_fe_coll is not null. Making FESpace and GridFunction." << std::endl;
      const int dims = nodal_fe_space->GetVDim();
      // Create a positive (Bernstein) grid function for the nodes
      mfem::FiniteElementSpace* pos_fe_space =
        new mfem::FiniteElementSpace(gf_mesh, pos_fe_coll, dims);
      mfem::GridFunction *pos_nodes = new mfem::GridFunction(pos_fe_space);

      // m_pos_nodes takes ownership of pos_fe_coll's memory (and pos_fe_space's memory)
      pos_nodes->MakeOwner(pos_fe_coll);

      // Project the nodal grid function onto this
      pos_nodes->ProjectGridFunction(*gf);

      out_pos_gf = pos_nodes;
      is_new = true;
    }
    //DEBUG
    else std::cerr << "BAD... pos_fe_coll is NULL. Could not make FESpace or GridFunction." << std::endl;

    //DEBUG
    if (!out_pos_gf)
    {
      std::cerr << "project_to_pos_basis(): Construction failed;  out_pos_gf is NULL!" << std::endl;
    }

  }

  return out_pos_gf;
}


}  // namespace dray
