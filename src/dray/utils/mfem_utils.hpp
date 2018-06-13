#ifndef DRAY_MFEM_UTILS_HPP
#define DRAY_MFEM_UTILS_HPP

#include <dray/mfem_mesh.hpp>

namespace dray
{
namespace detail
{

bool is_positive_basis(const mfem::FiniteElementCollection* fec);

/*!
 * \brief Utility function to get a positive (i.e. Bernstein)
 * collection of bases corresponding to the given FiniteElementCollection.
 *
 * \return A pointer to a newly allocated FiniteElementCollection
 * corresponding to \a fec
 * \note   It is the user's responsibility to deallocate this pointer.
 * \pre    \a fec is not already positive
 */
mfem::FiniteElementCollection* get_pos_fec(
  const mfem::FiniteElementCollection* fec,
  int order,
  int dim,
  int map_type);

} // namespace detail
} // namespace dray

#endif
