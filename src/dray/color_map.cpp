#include <dray/color_map.hpp>

namespace dray
{

ColorMap::ColorMap ()
: m_color_table ("cool2warm"), m_samples (1024), m_log_scale (false)
{
  m_color_table.sample (m_samples, m_colors);
}

void ColorMap::color_table (const ColorTable &color_table)
{
  m_color_table = color_table;
  m_color_table.sample (m_samples, m_colors);
}

void ColorMap::scalar_range (const Range<> &range)
{
  m_range = range;
}

void ColorMap::log_scale (bool on)
{
  m_log_scale = on;
}

void ColorMap::samples (int32 samples)
{
  assert (samples > 0);
  m_samples = samples;
  m_color_table.sample (m_samples, m_colors);
}

} // namespace dray
