#ifndef DRAY_MFEM_GRID_FUNCTION_HPP
#define DRAY_MFEM_GRID_FUNCTION_HPP

#include <dray/array.hpp>
#include <dray/range.hpp>
#include <dray/ray.hpp>
#include <dray/shading_context.hpp>
#include <dray/vec.hpp>
#include <mfem.hpp>
#include <memory>

namespace dray
{

/**
 * A wrapper around mfem::GridFunction that forces to use the Bernstein basis.
 */
class MFEMGridFunction
{
protected:
  mfem::GridFunction *m_pos_nodes;
  std::shared_ptr<mfem::GridFunction> _m_pos_nodes;
  Range m_range;

public:
  MFEMGridFunction();

  MFEMGridFunction(mfem::GridFunction *gf);
  ~MFEMGridFunction();

  Range get_field_range() const;

  //TODO I wish there were a way to return const * and enforce it.
  mfem::GridFunction *GetGridFunction() const { return m_pos_nodes; }

  template<typename T>
  Array<ShadingContext<T>> get_shading_context(Array<Ray<T>> &rays) const;


  template<typename T>
  void field_bounds(T &lower, T &upper, int32 comp = 1);

  void set_grid_function(mfem::GridFunction *gf);

  //template<typename T, int32 S>
  //void field_bounds(Vec<T,S> &lower, Vec<T,S> &upper) const;  //TODO
  void print_self();
};

} // namespace dray

#endif
