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
ElTransData<T,3> import_mesh(const mfem::Mesh &mfem_mesh);

template <typename T>
ElTransData<T,3> import_linear_mesh(const mfem::Mesh &mfem_mesh);

template <typename T>
ElTransData<T,3> import_grid_function_space(const mfem::GridFunction &mfem_gf);

template <typename T>
ElTransData<T,1> import_grid_function_field(const mfem::GridFunction &mfem_gf);


//
// Import MFEM data from MFEM file.
//

// TODO

}  // namespace dray

#endif // DRAY_MFEM2DRAY_HPP
