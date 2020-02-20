// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <dray/rendering/scalar_buffer.hpp>
#include <dray/error_check.hpp>
#include <dray/policies.hpp>
#include <dray/utils/png_encoder.hpp>

#include <conduit.hpp>
#include <conduit_relay.hpp>
#include <conduit_blueprint.hpp>

namespace dray
{

ScalarBuffer::ScalarBuffer ()
: m_width (1024),
  m_height (1024),
  m_clear_value(0.f)
{
  m_scalars.resize (m_width * m_height);
  m_depths.resize (m_width * m_height);
  clear ();
}

ScalarBuffer::ScalarBuffer (const int32 width, const int32 height)
: m_width (width),
  m_height (height),
  m_clear_value(0.f)
{
  assert (m_width > 0);
  assert (m_height > 0);
  m_scalars.resize (m_width * m_height);
  m_depths.resize (m_width * m_height);
  clear ();
}

int32 ScalarBuffer::width () const
{
  return m_width;
}

int32 ScalarBuffer::height () const
{
  return m_height;
}

void ScalarBuffer::save(const std::string name)
{
  conduit::Node mesh;
  mesh["coordsets/coords/type"] = "uniform";
  mesh["coordsets/coords/dims/i"] = m_width + 1;
  mesh["coordsets/coords/dims/j"] = m_height + 1;

  mesh["topologies/topo/coordset"] = "coords";
  mesh["topologies/topo/type"] = "uniform";

  mesh["fields/scalar/association"] = "element";
  mesh["fields/scalar/topology"] = "topo";
  const int size = m_scalars.size();
  const Float *scalars = m_scalars.get_host_ptr_const();
  mesh["fields/scalar/values"].set(scalars, size);

  mesh["fields/depth/association"] = "element";
  mesh["fields/depth/topology"] = "topo";
  const float32 *depths = m_depths.get_host_ptr_const();
  mesh["fields/depth/values"].set(depths, size);

  conduit::Node verify_info;
  bool ok = conduit::blueprint::mesh::verify(mesh,verify_info);
  if(!ok)
  {
    verify_info.print();
  }
  conduit::relay::io_blueprint::save(mesh, name + ".blueprint_root_hdf5");
}

void ScalarBuffer::clear()
{
  const int32 size = m_scalars.size();
  Float clear_value = m_clear_value;

  Float *scalar_ptr = m_scalars.get_device_ptr ();
  float32 *depth_ptr = m_depths.get_device_ptr ();

  RAJA::forall<for_policy> (RAJA::RangeSegment (0, size), [=] DRAY_LAMBDA (int32 ii) {
    depth_ptr[ii] = infinity<float32> ();
    scalar_ptr[ii] = clear_value;
  });
  DRAY_ERROR_CHECK();
}

} // namespace dray
