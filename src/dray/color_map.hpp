// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_COLOR_MAP_HPP
#define DRAY_COLOR_MAP_HPP

#include <dray/array.hpp>
#include <dray/color_table.hpp>
#include <dray/range.hpp>

namespace dray
{

/**
 * The ColorMap class encapsulates all the information
 * needed to convert a scalar to a color. This includes
 * the input scalar range, log scaling, color table
 * and possibly an out of range color.
 *
 */
class ColorMap
{
  protected:
  ColorTable m_color_table;
  Array<Vec<float32, 4>> m_colors;
  int32 m_samples; /*!< number of samples for the color table */
  Range<> m_range; /*!< scalar range to map to color */
  bool m_log_scale; /*!< log scale color lookup */
  public:
  ColorMap ();
  void color_table (const ColorTable &color_table);
  void scalar_range (const Range<> &range);
  void log_scale (bool on);
  void samples (int32 samples);

  friend class DeviceColorMap;
}; // class color map

} // namespace dray
#endif
