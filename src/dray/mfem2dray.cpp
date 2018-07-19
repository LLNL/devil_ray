#include <dray/mfem2dray.hpp>
#include <dray/policies.hpp>
#include <dray/types.hpp>

#include <iostream>

namespace dray
{

template <typename T>
ElTransData<T,3> import_mesh(const mfem::Mesh &mfem_mesh)
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
    std::cerr << "mfem2dray import_mesh() - GetNodes() is NOT null." << std::endl;
    return import_grid_function_space<T>(*mesh_nodes);
  }
  else
  {
    std::cerr << "mfem2dray import_mesh() - GetNodes() is NULL." << std::endl;
    return import_linear_mesh<T>(mfem_mesh);
  }
}

template <typename T>
ElTransData<T,3> import_linear_mesh(const mfem::Mesh &mfem_mesh)
{
  ElTransData<T,3> dataset;
  //TODO resize, import, etc.
  return dataset;
}

template <typename T>
ElTransData<T,3> import_grid_function_space(const mfem::GridFunction &mfem_gf)
{
  ElTransData<T,3> dataset;

  // Access to degree of freedom mapping.
  const mfem::FiniteElementSpace *fespace = mfem_gf.FESpace();

  // Access to control point data.
  mfem::Vector ctrl_vals;
  mfem_gf.GetTrueDofs(ctrl_vals);   // Sets size and initializes data. Might be reference.

  mfem::Array<int> zeroth_dof_set;
  fespace->GetElementDofs(0, zeroth_dof_set);

  const int32 dofs_per_element = zeroth_dof_set.Size();
  const int32 num_elements = fespace->GetNE();
  const int32 num_ctrls = ctrl_vals.Size();

  // Enforce: All elements must have same number of dofs.
  const int32 mfem_num_dofs = fespace->GetNDofs();

  std::cout<<"Mfem dofs "<<mfem_num_dofs<<" "<<num_elements * dofs_per_element<<"\n";
  std::cout<<"el 0 dof "<<dofs_per_element<<" nels "<<num_elements<<"\n";
  std::cout<<"num_ctrls "<<num_ctrls<<"\n";
  // I could be way off base here, but dofs could be shared between elements, so the number 
  // is lower than expected.
  assert(mfem_num_dofs == num_elements * dofs_per_element);

  dataset.resize(num_elements, dofs_per_element, num_ctrls);

  // Are these MFEM data structures thread-safe?  TODO

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
      ctrl_idx_ptr[dof_id] = el_dof_set[el_dof_id];
    }
  }
  ///});


  int32 stride_pdim;
  int32 stride_ctrl;
  if (fespace->GetOrdering() == mfem::Ordering::byNODES)  // XXXX YYYY ZZZZ
  {
    stride_pdim = num_elements;
    stride_ctrl = 1;
  }
  else                                                    // XYZ XYZ XYZ XYZ
  {
    stride_pdim = 1;
    stride_ctrl = 3;
  }

  //
  // Import degree of freedom values.
  //
  Vec<T,3> *ctrl_val_ptr = dataset.m_values.get_host_ptr();
  ///RAJA::forall<for_cpu_policy>(RAJA::RangeSegment(0, num_ctrls), [=] (int32 ctrl_id)
  for (int32 ctrl_id = 0; ctrl_id < num_ctrls; ctrl_id++)
  {
    // TODO get internal representation of the mfem memory, so we can access in a device function.
    //
    for (int32 pdim = 0; pdim < 3; pdim++)
      ctrl_val_ptr[ctrl_id][pdim] = ctrl_vals( pdim * stride_pdim + ctrl_id * stride_ctrl );
  }
  ///});

  return dataset;
}

template <typename T>
ElTransData<T,1> import_grid_function_field(const mfem::GridFunction &mfem_gf)
{
  ElTransData<T,1> dataset;
  //TODO resize, import, etc.
  return dataset;
}


// Explicit instantiations
template ElTransData<float32,3> import_mesh<float32>(const mfem::Mesh &mfem_mesh);
template ElTransData<float32,3> import_linear_mesh<float32>(const mfem::Mesh &mfem_mesh);
template ElTransData<float32,3> import_grid_function_space<float32>(const mfem::GridFunction &mfem_gf);
template ElTransData<float32,1> import_grid_function_field<float32>(const mfem::GridFunction &mfem_gf);

template ElTransData<float64,3> import_mesh<float64>(const mfem::Mesh &mfem_mesh);
template ElTransData<float64,3> import_linear_mesh<float64>(const mfem::Mesh &mfem_mesh);
template ElTransData<float64,3> import_grid_function_space<float64>(const mfem::GridFunction &mfem_gf);
template ElTransData<float64,1> import_grid_function_field<float64>(const mfem::GridFunction &mfem_gf);

}  // namespace dray
