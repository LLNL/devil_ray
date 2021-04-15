// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <dray/GridFunction/low_order_field.hpp>
#include <dray/policies.hpp>

namespace dray
{


Range LowOrderField::CalcRange::operator()(const LowOrderField * arg) const
{
  Range result;

  const Float *values_ptr = arg->m_values.get_device_ptr_const();
  const int32 size = arg->m_values.size();

  RAJA::ReduceMin<reduce_policy, Float> xmin (infinity<Float>());
  RAJA::ReduceMax<reduce_policy, Float> xmax (neg_infinity<Float>());

  RAJA::forall<for_policy> (RAJA::RangeSegment (0, size), [=] DRAY_LAMBDA (int32 ii)
  {
    const Float value = values_ptr[ii];
    xmin.min (value);
    xmax.max (value);
  });
  result.include (xmin.get ());
  result.include (xmax.get ());

  return result;
}


LowOrderField::LowOrderField(Array<Float> values, Assoc assoc)
  : m_assoc(assoc),
    m_values(values)
{

}

LowOrderField::~LowOrderField()
{

}

Array<Float>
LowOrderField::values()
{
  return m_values;
}

LowOrderField::Assoc
LowOrderField::assoc() const
{
  return m_assoc;
}

std::vector<Range> LowOrderField::range() const
{
  std::vector<Range> ranges;
  ranges.push_back(m_range.get());
  return ranges;
}

int32 LowOrderField::order() const
{
  if(m_assoc == Assoc::Vertex)
  {
    return 1;
  }
  else
  {
    return 0;
  }
}

std::string LowOrderField::type_name() const
{
  std::string name = "low_order_";
  if(m_assoc == Assoc::Vertex)
  {
    name += "vertex";
  }
  else
  {
    name += "element";
  }
  return name;
}

void LowOrderField::to_node(conduit::Node &n_field)
{
  n_field.reset();
  n_field["type_name"] = type_name();
  /// n_field["order"] = get_poly_order();

  throw std::logic_error(("Not implemented to_node()! " __FILE__));

  /// conduit::Node &n_gf = n_field["grid_function"];
  /// GridFunction<ElemT::get_ncomp ()> gf = get_dof_data();
  /// gf.to_node(n_gf);
}

void LowOrderField::to_blueprint(conduit::Node &n_dataset)
{
  conduit::Node &n_field = n_dataset["fields/" + m_name];

  // hard coded topology
  const std::string topo_name = "topo";
  n_field["topology"] = "topo";
  n_field["association"] = m_assoc == Assoc::Vertex ? "vertex" : "element";
  n_field["values"].set_external(m_values.get_host_ptr(), m_values.size());
}

} // namespace dray
