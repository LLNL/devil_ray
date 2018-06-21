#ifndef DRAY_MFEM_GRID_FUNCTION_HPP
#define DRAY_MFEM_GRID_FUNCTION_HPP

#include <dray/shading_context.hpp>
#include <dray/vec.hpp>
#include <dray/ray.hpp>
#include <dray/array.hpp>
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

  MFEMGridFunction() {}

public:
  MFEMGridFunction(mfem::GridFunction *gf);
  ~MFEMGridFunction();

      //TODO I wish there were a way to return const * and enforce it.
  mfem::GridFunction *GetGridFunction() const { return m_pos_nodes; }

  template<typename T>
  ShadingContext<T> get_shading_context(Ray<T> &rays) const;

  template<typename T>
  void field_bounds(T &lower, T &upper, int32 comp = 1) const;

  template<typename T, int32 S>
  void field_bounds(Vec<T,S> &lower, Vec<T,S> &upper) const;  //TODO

};

} // namespace dray

#endif
