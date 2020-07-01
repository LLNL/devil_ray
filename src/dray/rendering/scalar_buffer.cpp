// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <dray/rendering/scalar_buffer.hpp>

#include <dray/error_check.hpp>
#include <dray/policies.hpp>


#include <conduit.hpp>
#include <conduit_relay.hpp>
#include <conduit_blueprint.hpp>

namespace dray
{

namespace detail
{

template<typename FloatType>
void init_buffer(Array<FloatType> &scalars, const FloatType clear_value)
{
  const int32 size = scalars.size();
  FloatType *scalar_ptr = scalars.get_device_ptr ();

  RAJA::forall<for_policy> (RAJA::RangeSegment (0, size), [=] DRAY_LAMBDA (int32 ii) {
    scalar_ptr[ii] = clear_value;
  });
  DRAY_ERROR_CHECK();
}

} // namespace detail

bool ScalarBuffer::has_field(const std::string name)
{
  return m_scalars.find(name) != m_scalars.end();
}

void ScalarBuffer::add_field(const std::string name)
{
  Array<float32> scalar;
  scalar.resize(m_width * m_height);
  detail::init_buffer(scalar, m_clear_value);
  m_scalars[name] = scalar;
}

ScalarBuffer::ScalarBuffer()
 : m_width(0),
   m_height(0),
   m_clear_value(0)
{}

ScalarBuffer::ScalarBuffer(const int32 width,
                           const int32 height,
                           const float32 clear_value)
 : m_width(width),
   m_height(height),
   m_clear_value(clear_value)
{
  m_depths.resize(width * height);
  detail::init_buffer(m_depths, m_clear_value);
}

void ScalarBuffer::to_node(conduit::Node &mesh)
{
  mesh.reset();
  mesh["coordsets/coords/type"] = "uniform";
  mesh["coordsets/coords/dims/i"] = m_width + 1;
  mesh["coordsets/coords/dims/j"] = m_height + 1;

  mesh["topologies/topo/coordset"] = "coords";
  mesh["topologies/topo/type"] = "uniform";

  for(auto scalar : m_scalars)
  {
    const std::string path = "fields/" + scalar.first + "/";
    mesh[path + "association"] = "element";
    mesh[path + "topology"] = "topo";
    const int size = scalar.second.size();
    const float32 *scalars = scalar.second.get_host_ptr_const();
    mesh[path + "values"].set(scalars, size);
  }

  mesh["fields/depth/association"] = "element";
  mesh["fields/depth/topology"] = "topo";
  const int size = m_depths.size();
  const float32 *depths = m_depths.get_host_ptr_const();
  mesh["fields/depth/values"].set(depths, size);

  conduit::Node verify_info;
  bool ok = conduit::blueprint::mesh::verify(mesh,verify_info);
  if(!ok)
  {
    verify_info.print();
  }
}

} // namespace dray
