#ifndef DRAY_MFEM_VOLUME_INTEGRATOR_HPP
#define DRAY_MFEM_VOLUME_INTEGRATOR_HPP

#include <dray/mfem_data_set.hpp>
#include <dray/color_table.hpp>

namespace dray
{

class MFEMVolumeIntegrator
{
protected:
  MFEMMesh         m_mesh;
  MFEMGridFunction m_field;
  float32          m_sample_dist;
  ColorTable       m_color_table; 

  MFEMVolumeIntegrator(); 
public:
  MFEMVolumeIntegrator(MFEMMesh &mesh, MFEMGridFunction &gf); 
  ~MFEMVolumeIntegrator(); 
  void set_color_table(const ColorTable &color_table);

  template<typename T>
  Array<Vec<float32,4>> integrate(Ray<T> rays);
  
};

} // namespace dray

#endif
