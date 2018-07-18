#ifndef DRAY_MFEM2DRAY_HPP
#define DRAY_MFEM2DRAY_HPP

#include <mfem.hpp>

#include <dray/high_order_shape.hpp>  // For BernsteinShape<> and ElTrans<>.

namespace dray
{

/* Types */
template <typename T>
using HexShapeType = BernsteinShape<T,3>;      // Trivariate Bernstein-basis polynomials.

template <typename T>
using HexETS = ElTrans<T,3,3,HexShapeType<T>>;    // Element transformation for space.
template <typename T>
using HexETF = ElTrans<T,1,3,HexShapeType<T>>;    // Element transformation for field.


/* Functions */

//
// Import MFEM data from in-memory MFEM data structure.
//
template <typename T>
HexETS<T> import_mesh(const mfem::Mesh &mfem_mesh);

template <typename T>
HexETS<T> import_linear_mesh(const mfem::Mesh &mfem_mesh);

template <typename T>
HexETS<T> import_grid_function_space(const mfem::GridFunction &mfem_gf);

template <typename T>
HexETF<T> import_grid_function_field(const mfem::GridFunction &mfem_gf);


//
// Import MFEM data from MFEM file.
//

// TODO

}  // namespace dray

#endif // DRAY_MFEM2DRAY_HPP
