#ifndef DRAY_MFEM2DRAY_HPP
#define DRAY_MFEM2DRAY_HPP

#include <mfem.hpp>

#include <dray/high_order_shape.hpp>  // For BernsteinBasis<> and ElTransData<>.

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
ElTransData<T,3> import_mesh(const mfem::Mesh &mfem_mesh, int32 &space_P);

template <typename T>
ElTransData<T,3> import_linear_mesh(const mfem::Mesh &mfem_mesh);

template <typename T, int32 PhysDim>
ElTransData<T,PhysDim> import_grid_function(const mfem::GridFunction &mfem_gf, int32 &field_P);


//
// Import MFEM data from MFEM file.
//

// TODO

}  // namespace dray

#endif // DRAY_MFEM2DRAY_HPP
