#ifndef DRAY_MFEM_GRID_FUNCTION_HPP
#define DRAY_MFEM_GRID_FUNCTION_HPP

#include <dray/vec.hpp>
#include <mfem.hpp>

namespace dray
{

/**
 * A wrapper around mfem::GridFunction that forces to use the Bernstein basis.
 */
class MFEMGridFunction
{
protected:
  mfem::GridFunction *m_pos_nodes;
  bool m_delete_nodes;

  MFEMGridFunction() {}

public:
  MFEMGridFunction(mfem::GridFunction *gf);
  ~MFEMGridFunction() { if (m_delete_nodes) delete m_pos_nodes; }

      //TODO I wish there were a way to return const * and enforce it.
  mfem::GridFunction *GetGridFunction() { return m_pos_nodes; }

  template<typename T>
  void get_bounds(T &lower, T &upper, int32 comp = 1);

  template<typename T, int32 S>
  void get_bounds(Vec<T,S> &lower, Vec<T,S> &upper);  //TODO

};

} // namespace dray

#endif
