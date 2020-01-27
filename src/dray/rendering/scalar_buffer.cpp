// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <dray/rendering/scalar_buffer.hpp>
#include <dray/policies.hpp>
#include <dray/utils/png_encoder.hpp>

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
//
//void ScalarBuffer::save (const std::string name)
//{
//  PNGEncoder png_encoder;
//
//  png_encoder.encode ((float *)m_colors.get_host_ptr (), m_width, m_height);
//
//  png_encoder.save (name + ".png");
//}
//
//void ScalarBuffer::save_depth (const std::string name)
//{
//
//  int32 image_size = m_width * m_height;
//
//  const float32 *depth_ptr = m_depths.get_device_ptr_const ();
//
//  RAJA::ReduceMin<reduce_policy, float32> min_val (infinity32 ());
//  RAJA::ReduceMax<reduce_policy, float32> max_val (neg_infinity32 ());
//
//  RAJA::forall<for_policy> (RAJA::RangeSegment (0, image_size), [=] DRAY_LAMBDA (int32 i) {
//    const float32 depth = depth_ptr[i];
//    if (depth != infinity32 ())
//    {
//      min_val.min (depth);
//      max_val.max (depth);
//    }
//  });
//
//  float32 minv = min_val.get ();
//  float32 maxv = max_val.get ();
//  const float32 len = maxv - minv;
//
//  Array<float32> dbuffer;
//  dbuffer.resize (image_size * 4);
//
//  float32 *d_ptr = dbuffer.get_host_ptr ();
//
//  RAJA::forall<for_policy> (RAJA::RangeSegment (0, image_size), [=] DRAY_LAMBDA (int32 i) {
//    const float32 depth = depth_ptr[i];
//    float32 value = 0.f;
//
//    if (depth != infinity32 ())
//    {
//      value = (depth - minv) / len;
//    }
//    const int32 offset = i * 4;
//    d_ptr[offset + 0] = value;
//    d_ptr[offset + 1] = value;
//    d_ptr[offset + 2] = value;
//    d_ptr[offset + 3] = 1.f;
//  });
//
//  PNGEncoder png_encoder;
//
//  png_encoder.encode (dbuffer.get_host_ptr (), m_width, m_height);
//
//  png_encoder.save (name + ".png");
//}

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
}

} // namespace dray
