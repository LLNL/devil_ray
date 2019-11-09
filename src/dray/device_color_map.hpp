#ifndef DRAY_DEVICE_COLOR_MAP_HPP
#define DRAY_DEVICE_COLOR_MAP_HPP

#include <dray/color_map.hpp>
#include <dray/error.hpp>

namespace dray
{

/**
 * The DeviceColorMap class is a device accessible
 * color map.
 *
 */
class DeviceColorMap
{
public:
  const Vec<float32,4> * m_colors;
  const int32 m_size;
  const bool m_log_scale;

  Float m_inv_range;
  Float m_min;

  DeviceColorMap() = delete;

  DeviceColorMap(ColorMap &color_map)
    : m_colors(color_map.m_colors.get_device_ptr_const()),
      m_size(color_map.m_colors.size()),
      m_log_scale(m_log_scale)
  {
    if(color_map.m_range.is_empty())
    {
      throw DRayError("ColorMap scalar range never set");
    }

    if(!m_log_scale)
    {
      m_min = color_map.m_range.min();
      m_inv_range = rcp_safe( color_map.m_range.length() );
    }
    else
    {
      // log scale. do some checking
      Range<> range = color_map.m_range;
      if(range.min() <= 0.f)
      {
        throw DRayError("DeviceColorMap log scalar range contains values <= 0");
      }
      m_min = log(color_map.m_range.min());
      m_inv_range = rcp_safe( log(color_map.m_range.length()) );
    }
  }

  DRAY_EXEC Vec<float32,4> color(const Float &scalar) const
  {
    Float s = scalar;
    if(m_log_scale)
    {
      s = log(scalar);
    }

    const float32 normalized = static_cast<float32>((s - m_min) * m_inv_range);
    int32 sample_idx = static_cast<int32>(normalized * float32(m_size - 1));
    sample_idx = clamp(sample_idx, 0, m_size - 1);
    return m_colors[sample_idx];
  }
}; // class device color map

} // namespace dray
#endif
