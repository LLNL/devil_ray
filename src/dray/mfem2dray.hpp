#ifndef DRAY_MFEM2DRAY_HPP
#define DRAY_MFEM2DRAY_HPP

#include <mfem.hpp>

#include <dray/GridFunction/field.hpp>
#include <dray/GridFunction/grid_function_data.hpp>
#include <dray/GridFunction/mesh.hpp>

namespace dray
{

/* Types */
using BernsteinHex = BernsteinBasis<3>; // Trivariate Bernstein-basis polynomials.


/* Functions */

// TODO import_mesh() needs to know the element type to ask for the correct basis conversion.

//
// Import MFEM data from in-memory MFEM data structure.
//
GridFunctionData<3> import_mesh (const mfem::Mesh &mfem_mesh, int32 &space_P);

GridFunctionData<3> import_linear_mesh (const mfem::Mesh &mfem_mesh);

template <int32 PhysDim>
GridFunctionData<PhysDim>
import_grid_function (const mfem::GridFunction &mfem_gf, int32 &field_P);

GridFunctionData<1>
import_vector_field_component (const mfem::GridFunction &_mfem_gf, int32 comp, int32 &field_P);


//
// Get dray::Mesh or dray::Field.
//

template <class ElemT> Mesh<ElemT> import_mesh (const mfem::Mesh &mfem_mesh);

template <class ElemT, uint32 ncomp = 1>
Field<FieldOn<ElemT, ncomp>> import_field (const mfem::GridFunction &mfem_gf);

template <class ElemT>
Field<FieldOn<ElemT, 1>>
import_vector_field_component (const mfem::GridFunction &mfem_gf, int32 comp);


//
// project_to_pos_basis()
//
// Helper function prototype.
// If is_new was set to true, the caller is responsible for deleting the returned pointer.
// If is_new was set to false, then the returned value is null, and the caller should use gf.
mfem::GridFunction *project_to_pos_basis (const mfem::GridFunction *gf, bool &is_new);


//
// Import MFEM data from MFEM file.
//

// TODO

} // namespace dray

#endif // DRAY_MFEM2DRAY_HPP
