#ifndef DRAY_MFEM2DRAY_HPP
#define DRAY_MFEM2DRAY_HPP

#include <mfem.hpp>

#include <dray/high_order_shape.hpp>  // For BernsteinBasis<>
#include <dray/GridFunction/grid_function_data.hpp>
#include <dray/GridFunction/mesh.hpp>
#include <dray/GridFunction/field.hpp>

namespace dray
{

/* Types */
template <typename T>
using BernsteinHex = BernsteinBasis<T,3>;      // Trivariate Bernstein-basis polynomials.


/* Functions */

//
// Import MFEM data from in-memory MFEM data structure.
//
template <typename T>
GridFunctionData<T,3> import_mesh(const mfem::Mesh &mfem_mesh, int32 &space_P);

template <typename T>
GridFunctionData<T,3> import_linear_mesh(const mfem::Mesh &mfem_mesh);

template <typename T, int32 PhysDim>
GridFunctionData<T,PhysDim> import_grid_function(const mfem::GridFunction &mfem_gf, int32 &field_P);

template <typename T>
GridFunctionData<T,1> import_vector_field_component(const mfem::GridFunction &_mfem_gf, int32 comp, int32 &field_P);


//
// Get dray::Mesh or dray::Field.
//

template <typename T>
Mesh<T,3> import_mesh(const mfem::Mesh &mfem_mesh);

template <typename T, int32 PhysDim = 1>
Field<T,3,PhysDim> import_field(const mfem::GridFunction &mfem_gf);

template <typename T>
Field<T,3,1> import_vector_field_component(const mfem::GridFunction &mfem_gf, int32 comp);



//
// Import MFEM data from MFEM file.
//

// TODO

}  // namespace dray

#endif // DRAY_MFEM2DRAY_HPP
