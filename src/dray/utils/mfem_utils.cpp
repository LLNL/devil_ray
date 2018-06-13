#include <dray/utils/mfem_utils.hpp>

namespace dray
{
namespace detail
{

bool is_positive_basis(const mfem::FiniteElementCollection* fec)
{
  // HACK: Check against several common expected FE types

  if(fec == nullptr)
  {
    return false;
  }

  if(const mfem::H1_FECollection* h1Fec =
       dynamic_cast<const mfem::H1_FECollection*>(fec))
  {
    return h1Fec->GetBasisType() == mfem::BasisType::Positive;
  }

  if(const mfem::L2_FECollection* l2Fec =
       dynamic_cast<const mfem::L2_FECollection*>(fec))
  {
    return l2Fec->GetBasisType() == mfem::BasisType::Positive;
  }

  if( dynamic_cast<const mfem::NURBSFECollection*>(fec)       ||
      dynamic_cast<const mfem::LinearFECollection*>(fec)      ||
      dynamic_cast<const mfem::QuadraticPosFECollection*>(fec) )
  {
    return true;
  }

  return false;
}

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
  int map_type)
{
  //SLIC_CHECK_MSG( !isPositiveBasis( fec),
  //                "This function is only meant to be called "
  //                "on non-positive finite element collection" );

  // Attempt to find the corresponding positive H1 fec
  if(dynamic_cast<const mfem::H1_FECollection*>(fec))
  {
    return new mfem::H1_FECollection(order, dim, mfem::BasisType::Positive);
  }

  // Attempt to find the corresponding positive L2 fec
  if(dynamic_cast<const mfem::L2_FECollection*>(fec))
  {
    // should we throw a not supported error here?
    return new mfem::L2_FECollection(order, dim, mfem::BasisType::Positive,
                                     map_type);
  }

  // Attempt to find the corresponding quadratic or cubic fec
  // Note: Linear FECollections are positive
  if(dynamic_cast<const mfem::QuadraticFECollection*>(fec) ||
     dynamic_cast<const mfem::CubicFECollection*>(fec) )
  {
    //SLIC_ASSERT( order == 2 || order == 3);
    return new mfem::H1_FECollection(order, dim, mfem::BasisType::Positive);
  }

  // Give up -- return NULL
  return nullptr;
}


} // namespace detail
} // namespace dray
