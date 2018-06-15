#ifndef DRAY_MFEM_GRID_FUNCTION_HPP
#define DRAY_MFEM_GRID_FUNCTION_HPP

#include <dray/vec.hpp>
#include <dray/ray.hpp>
#include <dray/array.hpp>
#include <mfem.hpp>
#include <memory>

namespace dray
{

//TODO make a new class for shading context

template<typename T>
class ShadingContext
{
public:
  //Array<int32>    m_is_valid;
  //Array<Vec<T,3>> m_hit_pt;
  Array<Vec<T,3>> m_normal;
  Array<T>  m_sample_val;
  //Array<Vec<T,3>> m_ray_dir;
  //Array<int32>    m_pixel_id;

  void resize(const int32 size)
  {
    //m_is_valid.resize(size);
    //m_hit_pt.resize(size);
    m_normal.resize(size);
    m_sample_val.resize(size);
    //m_ray_dir.resize(size);
    //m_pixel_id.resize(size);
  }

  int32 size() const { return m_normal.size(); }
};



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
  mfem::GridFunction *GetGridFunction() { return m_pos_nodes; }

  template<typename T>
  void get_shading_context(const Ray<T> &rays, ShadingContext<T> &shading_ctx) const;

  template<typename T>
  void field_bounds(T &lower, T &upper, int32 comp = 1) const;

  template<typename T, int32 S>
  void field_bounds(Vec<T,S> &lower, Vec<T,S> &upper) const;  //TODO

};

} // namespace dray

#endif
