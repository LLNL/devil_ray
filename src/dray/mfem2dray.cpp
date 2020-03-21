// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <dray/GridFunction/field.hpp>
#include <dray/GridFunction/mesh.hpp>
#include <dray/mfem2dray.hpp>
#include <dray/policies.hpp>
#include <dray/error.hpp>
#include <dray/error_check.hpp>
#include <dray/types.hpp>
#include <dray/utils/mfem_utils.hpp>
#include <dray/utils/data_logger.hpp>

#include <dray/derived_topology.hpp>

#include <iostream>

namespace dray
{

namespace detail
{

// Assumes that values is allocated
template <int32 PhysDim, int32 RefDim>
void
import_dofs(const mfem::GridFunction &mfem_gf,
            Array<Vec<Float, PhysDim>> &values,
            int comp)
{
  // Access to degree of freedom mapping.
  const mfem::FiniteElementSpace *fespace = mfem_gf.FESpace ();

  // Access to control point data.
  const mfem::Vector &ctrl_vals = mfem_gf;

  const int32 P = fespace->GetOrder (0);

  mfem::Array<int> zeroth_dof_set;
  fespace->GetElementDofs (0, zeroth_dof_set);

  const int32 vdim = fespace->GetVDim();
  const int32 dofs_per_element = zeroth_dof_set.Size ();
  const int32 num_elements = fespace->GetNE ();
  const int32 num_ctrls = ctrl_vals.Size () / vdim;

  mfem::Table el_dof_table (fespace->GetElementToDofTable ());
  el_dof_table.Finalize ();
  const int32 all_el_dofs = el_dof_table.Size_of_connections ();
  if(all_el_dofs != num_elements * dofs_per_element)
  {
    DRAY_ERROR("Elements do not have the same number of dofs");
  }

  // if this is a vector or points in 2d,
  // we will convert this into 3d space with 0 for z;
  bool fill_z = false;
  if(RefDim == 2 && vdim > 1 && PhysDim == 3)
  {
    fill_z = true;
  }

  // Former attempt at the above assertion.
  const int32 mfem_num_dofs = fespace->GetNDofs ();

  int32 stride_pdim;
  int32 stride_ctrl;
  if (fespace->GetOrdering () == mfem::Ordering::byNODES) // XXXX YYYY ZZZZ
  {
    DRAY_INFO("Ordering by nodes\n");
    // stride_pdim = num_elements;
    stride_pdim = num_ctrls;
    stride_ctrl = 1;
  }
  else // XYZ XYZ XYZ XYZ
  {
    DRAY_INFO("Ordering interleaved\n");
    stride_pdim = 1;
    stride_ctrl = vdim;
  }

  //
  // Import degree of freedom values.
  //
  Vec<Float, PhysDim> *ctrl_val_ptr = values.get_host_ptr();
  // import all components
  if(comp == -1)
  {
    /// RAJA::forall<for_cpu_policy>(RAJA::RangeSegment(0, num_ctrls), [=] (int32 ctrl_id)
    for (int32 ctrl_id = 0; ctrl_id < num_ctrls; ctrl_id++)
    {
      // TODO get internal representation of the mfem memory, so we can access in a device function.
      //
      for (int32 pdim = 0; pdim < PhysDim; pdim++)
      {
        if(fill_z && pdim == 2)
        {
          ctrl_val_ptr[ctrl_id][pdim] = Float(0.f);
        }
        else
        {
          ctrl_val_ptr[ctrl_id][pdim] = ctrl_vals (pdim * stride_pdim + ctrl_id * stride_ctrl);
        }
      }
    }
    ///});
    DRAY_ERROR_CHECK();
  }
  else
  {
    if(comp >= vdim)
    {
      DRAY_ERROR("vector dim is greater then requested component");
    }
    //import only a single component
    for (int32 ctrl_id = 0; ctrl_id < num_ctrls; ctrl_id++)
    {
      ctrl_val_ptr[ctrl_id][0] = ctrl_vals (comp * stride_pdim + ctrl_id * stride_ctrl);
    }
  }
}

// Assumes that values is allocated
template <int32 PhysDim, int32 RefDim>
void
import_indices(const mfem::GridFunction &mfem_gf,
               Array<int32> &indexs)
{
  // Access to degree of freedom mapping.
  const mfem::FiniteElementSpace *fespace = mfem_gf.FESpace ();

  mfem::Array<int> zeroth_dof_set;
  fespace->GetElementDofs (0, zeroth_dof_set);

  const int32 P = fespace->GetOrder (0);
  const int32 num_elements = fespace->GetNE ();
  const int32 dofs_per_element = zeroth_dof_set.Size ();

  // DRAY and MFEM may store degrees of freedom in different orderings.
  bool use_dof_map = fespace->Conforming ();

  mfem::Array<int> fe_dof_map;
  // figure out what kinds of elements these are
  std::string elem_type(fespace->FEColl()->Name());

  if(elem_type.find("H1Pos") != std::string::npos)
  {
    if(RefDim == 3)
    {
      mfem::H1Pos_HexahedronElement h1_prototype (P);
      fe_dof_map = h1_prototype.GetDofMap();
    }
    else
    {
      mfem::H1Pos_QuadrilateralElement h1_prototype (P);
      fe_dof_map = h1_prototype.GetDofMap();
    }
  }
  else
  {
    // The L2 prototype does not return anything, because
    // the ording is implicit. Like somehow I was just supposed
    // to know that and should have expected an empty array.
    // Going to make the assumption that this is just a linear ordering.
    //mfem::L2Pos_HexahedronElement l2_prototype(P);
    use_dof_map = false;
  }

  int32 *ctrl_idx_ptr = indexs.get_host_ptr ();
  for (int32 el_id = 0; el_id < num_elements; el_id++)
  {
    // TODO get internal representation of the mfem memory, so we can access in a device function.
    //
    mfem::Array<int> el_dof_set;
    fespace->GetElementDofs (el_id, el_dof_set);
    int dof_size = el_dof_set.Size ();

    for (int32 dof_id = el_id * dofs_per_element, el_dof_id = 0;
         el_dof_id < dofs_per_element; dof_id++, el_dof_id++)
    {
      // Maintain same lexicographic order as MFEM (X-inner:Z-outer).
      const int32 el_dof_id_lex = el_dof_id;
      // Maybe there's a better practice than this inner conditional.
      const int32 mfem_el_dof_id = use_dof_map ? fe_dof_map[el_dof_id_lex] : el_dof_id_lex;
      ctrl_idx_ptr[dof_id] = el_dof_set[mfem_el_dof_id];
    }
  }
}

} // namespace detail



template <int32 PhysDim, int32 RefDim>
GridFunction<PhysDim>
import_grid_function2(const mfem::GridFunction &_mfem_gf, int32 &space_P, int comp = -1)
{
  bool is_gf_new;
  mfem::GridFunction *pos_gf = project_to_pos_basis (&_mfem_gf, is_gf_new);
  const mfem::GridFunction &mfem_gf = (is_gf_new ? *pos_gf : _mfem_gf);

  constexpr int32 phys_dim = PhysDim;
  GridFunction<phys_dim> grid_func;

  // Access to degree of freedom mapping.
  const mfem::FiniteElementSpace *fespace = mfem_gf.FESpace ();

  const int32 P = fespace->GetOrder (0);

  mfem::Array<int> zeroth_dof_set;
  fespace->GetElementDofs (0, zeroth_dof_set);

  const int32 vdim = fespace->GetVDim();
  const int32 dofs_per_element = zeroth_dof_set.Size ();
  const int32 num_elements = fespace->GetNE ();
  const int32 num_ctrls = mfem_gf.Size () / vdim;

  grid_func.resize (num_elements, dofs_per_element, num_ctrls);

  detail::import_dofs<PhysDim,RefDim>(mfem_gf, grid_func.m_values, comp);
  detail::import_indices<PhysDim,RefDim>(mfem_gf, grid_func.m_ctrl_idx);

  if (is_gf_new)
  {
    delete pos_gf;
  }

  space_P = P;
  return grid_func;
}

void import_field(DataSet &dataset,
                  const mfem::GridFunction &grid_function,
                  const mfem::Geometry::Type geom_type,
                  const std::string field_name,
                  const int32 comp) // single componet of vector (-1 all)
{

  if(geom_type != mfem::Geometry::CUBE && geom_type != mfem::Geometry::SQUARE)
  {
    DRAY_ERROR("Only hex and quad imports implemented");
  }

  int ref_dim = 3;
  if(geom_type == mfem::Geometry::SQUARE)
  {
    ref_dim = 2;
  }

  if(ref_dim == 3)
  {
    using HexScalar  = Element<3u, 1u, ElemType::Quad, Order::General>;
    int order;
    GridFunction<1> field_data = import_grid_function2<1,3> (grid_function, order, comp);
    Field<HexScalar> field (field_data, order, field_name);
    dataset.add_field(std::make_shared<Field<HexScalar>>(field));
  }
  else
  {
    using QuadScalar  = Element<2u, 1u, ElemType::Quad, Order::General>;
    int order;
    GridFunction<1> field_data = import_grid_function2<1,2> (grid_function, order, comp);
    Field<QuadScalar> field (field_data, order, field_name);
    dataset.add_field(std::make_shared<Field<QuadScalar>>(field));
  }

}

DataSet import_mesh(const mfem::Mesh &mesh)
{
  mfem::Geometry::Type geom_type = mesh.GetElementBaseGeometry(0);

  if(geom_type != mfem::Geometry::CUBE && geom_type != mfem::Geometry::SQUARE)
  {
    DRAY_ERROR("Only hex and quad imports implemented");
  }

  int ref_dim = 3;
  if(geom_type == mfem::Geometry::SQUARE)
  {
    ref_dim = 2;
  }

  mesh.GetNodes();

  if (mesh.Conforming())
  {
    DRAY_INFO("Conforming mesh");
  }
  else
  {
    DRAY_INFO("Non-Conforming mesh");
  }

  const mfem::GridFunction *nodes = mesh.GetNodes();;

  DataSet res;
  if (nodes != NULL)
  {
    if(ref_dim == 3)
    {
      using HexMesh = MeshElem<3u, Quad, General>;
      int order;
      GridFunction<3> gf = import_grid_function2<3,3> (*nodes, order);
      Mesh<HexMesh> mesh (gf, order);
      std::shared_ptr<HexTopology> topo = std::make_shared<HexTopology>(mesh);
      DataSet dataset(topo);
      res = dataset;
    }
    else
    {
      using QuadMesh = MeshElem<2u, Quad, General>;
      int order;
      GridFunction<3> gf = import_grid_function2<3,2> (*nodes, order);
      Mesh<QuadMesh> mesh (gf, order);
      std::shared_ptr<QuadTopology> topo = std::make_shared<QuadTopology>(mesh);
      DataSet dataset(topo);
      res = dataset;
    }
  }
  else
  {
    DRAY_ERROR("Importing linear mesh not implemented");
    //space_P = 1;
    //return import_linear_mesh (mfem_mesh);
  }

  return res;
}


void print_geom(mfem::Geometry::Type type)
{
  if(type == mfem::Geometry::POINT)
  {
    std::cout<<"point\n";
  }
  else if(type == mfem::Geometry::SEGMENT)
  {
    std::cout<<"segment\n";
  }
  else if(type == mfem::Geometry::TRIANGLE)
  {
    std::cout<<"triangle\n";
  }
  else if(type == mfem::Geometry::TETRAHEDRON)
  {
    std::cout<<"tet\n";
  }
  else if(type == mfem::Geometry::SQUARE)
  {
    std::cout<<"quad\n";
  }
  else if(type == mfem::Geometry::CUBE)
  {
    std::cout<<"hex\n";
  }
  else if(type == mfem::Geometry::PRISM)
  {
    std::cout<<"prism. no thanks\n";
  }
  else
  {
    std::cout<<"unknown\n";
  }
}

//
// project_to_pos_basis()
//
// If is_new was set to true, the caller is responsible for deleting the returned pointer.
// If is_new was set to false, then the returned value is null, and the caller should use gf.
mfem::GridFunction *project_to_pos_basis (const mfem::GridFunction *gf, bool &is_new)
{
  mfem::GridFunction *out_pos_gf = nullptr;
  is_new = false;

  /// bool is_high_order =
  ///    (gf != nullptr) && (mesh->GetNE() > 0);
  /// if(!is_high_order) std::cout<<"NOT High Order\n";

  // Sanity checks
  /// assert(is_high_order);
  assert (gf != nullptr);

  /// Generate (or access existing) positive (Bernstein) nodal grid function
  const mfem::FiniteElementSpace *nodal_fe_space = gf->FESpace ();
  if (nodal_fe_space == nullptr)
  {
    DRAY_ERROR("project_to_pos_basis(): nodal_fe_space is NULL!");
  }

  const mfem::FiniteElementCollection *nodal_fe_coll = nodal_fe_space->FEColl ();
  if (nodal_fe_coll == nullptr)
  {
    DRAY_ERROR("project_to_pos_basis(): nodal_fe_coll is NULL!");
  }

  // Check if grid function is positive, if not create positive grid function
  if (detail::is_positive_basis (nodal_fe_coll))
  {
    // std::cerr<<"Already positive.\n";
    is_new = false;
    out_pos_gf = nullptr;
  }
  else
  {
    // std::cerr<<"Attemping to convert to positive basis.\n";
    // Assume that all elements of the mesh have the same order and geom type
    mfem::Mesh *gf_mesh = nodal_fe_space->GetMesh ();
    if (gf_mesh == nullptr)
    {
      DRAY_ERROR("project_to_pos_basis(): gf_mesh is NULL!");
    }

    int order = nodal_fe_space->GetOrder (0);
    int dim = gf_mesh->Dimension ();
    mfem::Geometry::Type geom_type = gf_mesh->GetElementBaseGeometry (0);
    int map_type = (nodal_fe_coll != nullptr) ?
                   nodal_fe_coll->FiniteElementForGeometry (geom_type)->GetMapType () :
                   static_cast<int> (mfem::FiniteElement::VALUE);

    mfem::FiniteElementCollection *pos_fe_coll =
    detail::get_pos_fec (nodal_fe_coll, order, dim, map_type);

    if (pos_fe_coll != nullptr)
    {
      const int dims = nodal_fe_space->GetVDim ();
      // Create a positive (Bernstein) grid function for the nodes
      mfem::FiniteElementSpace *pos_fe_space =
      new mfem::FiniteElementSpace (gf_mesh, pos_fe_coll, dims);
      mfem::GridFunction *pos_nodes = new mfem::GridFunction (pos_fe_space);

      // m_pos_nodes takes ownership of pos_fe_coll's memory (and pos_fe_space's memory)
      pos_nodes->MakeOwner (pos_fe_coll);

      // Project the nodal grid function onto this
      pos_nodes->ProjectGridFunction (*gf);

      out_pos_gf = pos_nodes;
      is_new = true;
    }
    // DEBUG
    else
    {
      DRAY_ERROR("BAD... pos_fe_coll is NULL. Could not make FESpace or GridFunction.");
    }
    // DEBUG
    if (!out_pos_gf)
    {
      DRAY_ERROR("project_to_pos_basis(): Construction failed;  out_pos_gf is NULL!");
    }
  }

  return out_pos_gf;
}


} // namespace dray
