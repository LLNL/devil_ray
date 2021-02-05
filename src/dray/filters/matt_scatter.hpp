#ifndef DRAY_UNCOLLIDED_FLUX_HPP
#define DRAY_UNCOLLIDED_FLUX_HPP

#include <dray/data_set.hpp>
#include <dray/collection.hpp>
#include <dray/uniform_topology.hpp>
#include <dray/ray.hpp>
#include <dray/GridFunction/low_order_field.hpp>

namespace dray
{

struct DomainData
{
  UniformTopology *m_topo;
  LowOrderField *m_cross_section;
  LowOrderField *m_source;
}; // domain data

struct UniformData
{
  Vec<Float,3> m_spacing;
  Vec<Float,3> m_origin;
  Vec<int32,3> m_dims;
  int32 m_rank;
};

class UncollidedFlux
{
public:

protected:
  std::string m_total_cross_section_field;
  std::string m_emission_field;
  std::string m_overwrite_first_scatter_field;
  int32 m_legendre_order;
  std::vector<DomainData> m_domain_data;
  std::vector<UniformData> m_global_coords;
  int32 m_num_groups;

  // hack
  Float m_sigs;
public:
  UncollidedFlux();
  void execute(DataSet &data_set);
  void execute(Collection &collection);
  void domain_data(Collection &collection);

  // Absorption
  void total_cross_section_field(const std::string field_name);

  Array<Ray> create_rays(Array<Vec<Float,3>> sources);

  // Emission (original source)
  void emission_field(const std::string field_name);

  // Result of first scatter. Can be same as emission.
  void overwrite_first_scatter_field(const std::string field_name);

  int32 legendre_order() const;
  void legendre_order(int32 l_order);

  // Hack. TODO import and use SigmaS matrix variable.
  void uniform_isotropic_scattering(Float sigs);

};

};//namespace dray

#endif //DRAY_UNCOLLIDED_FLUX_HPP
