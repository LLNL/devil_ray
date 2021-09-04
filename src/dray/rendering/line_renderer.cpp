// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <dray/rendering/line_renderer.hpp>
#include <dray/math.hpp>
#include <dray/dray.hpp>
#include <dray/error.hpp>
#include <dray/error_check.hpp>
#include <dray/policies.hpp>
#include <dray/utils/timer.hpp>
#include <dray/rendering/rasterbuffer.hpp>

namespace dray
{
void crop_line_to_bounds(Vec<int32, 2> &p1, Vec<int32, 2> &p2, int32 width, int32 height)
{
  bool p1_ok, p2_ok;
  p1_ok = p2_ok = false;

  float32 x1, y1, x2, y2;
  x1 = p1[0];
  y1 = p1[1];
  x2 = p2[0];
  y2 = p2[1];

  // check that out points are within bounds
  if (x1 > -1 && x1 < width && y1 > -1 && y1 < height)
  {
    p1_ok = true;
  }
  if (x2 > -1 && x2 < width && y2 > -1 && y2 < height)
  {
    p2_ok = true;
  }

  // if both points are within bounds then there is nothing further to do
  if (p1_ok && p2_ok)
  {
    return;
  }

  // calculate the equation of the line
  float32 m = (y2 - y1) / (x2 - x1);
  float32 b = y1 - m * x1;

  const int32 top = 0;
  const int32 bottom = 1;
  const int32 left = 2;
  const int32 right = 3;

  int32 intersections_within_bounds[4];
  Vec<float32, 2> intersect_coords[4];

  // calculate the intersection points for each of the 4 sides
  intersect_coords[top] = {{((height - 1) - b) / m, (float32) (height - 1)}};
  intersect_coords[bottom] = {{(-1 * b) / m, 0}};
  intersect_coords[left] = {{0, b}};
  intersect_coords[right] = {{(float32) (width - 1), m * (width - 1) + b}};
  // determine which of the intersection points are within bounds
  bool none_within_bounds = true;
  for (int32 i = 0; i < 4; i ++)
  {
    if (intersect_coords[i][0] > -1 && intersect_coords[i][0] < width &&
        intersect_coords[i][1] > -1 && intersect_coords[i][1] < height)
    {
      intersections_within_bounds[i] = true;
      none_within_bounds = false;
    }
    else
    {
      intersections_within_bounds[i] = false;
    }
  }

  // if our line never passes across screen space
  if (none_within_bounds)
  {
    if (p1_ok || p2_ok)
    {
      fprintf(stderr, "line cropping has determined that the current line never crosses the screen, yet at least one of the endpoints is simultaneously on the screen, which is a contradiction.\n");
      exit(1);
    }
    p1[0] = -1;
    p1[1] = -1;
    p2[0] = -1;
    p2[1] = -1;
    return;
  }

  // tie breaking - make sure that a maximum of two sides are marked as having valid intersections
  if (intersections_within_bounds[top] && intersections_within_bounds[bottom])
  {
    intersections_within_bounds[left] = false;
    intersections_within_bounds[right] = false;
  }
  if (intersections_within_bounds[right] && intersections_within_bounds[left])
  {
    intersections_within_bounds[top] = false;
    intersections_within_bounds[bottom] = false;
  }

  // next we set up a data structure to house information about our intersections
  // only two intersections will actually be in view of the camera
  // so for each of the two intersections, we record distance^2 to p1, 
  // and the x and y vals of the intersection point
  float32 intersection_info[6];

  int32 index = 0;
  for (int32 i = 0; i < 4; i ++)
  {
    if (intersections_within_bounds[i])
    {
      float32 y1_minus_newy = y1 - intersect_coords[i][1];
      float32 x1_minus_newx = x1 - intersect_coords[i][0];
      // the first three spots are for one intersection
      intersection_info[index + 0] = (int32) (y1_minus_newy * y1_minus_newy + x1_minus_newx * x1_minus_newx);
      intersection_info[index + 1] = intersect_coords[i][0];
      intersection_info[index + 2] = intersect_coords[i][1];
      // then we increment by 3 to get to the next three spots
      index += 3;
    }
  }

  // with this information we can assign new values to p1 and p2 if needed
  float32 distance1 = intersection_info[0];
  float32 distance2 = intersection_info[3];
  index = distance1 < distance2 ? 0 : 1;
  if (!p1_ok)
  {
    p1[0] = intersection_info[index * 3 + 1];
    p1[1] = intersection_info[index * 3 + 2];
  }

  index = !index;
  if (!p2_ok)
  {
    p2[0] = intersection_info[index * 3 + 1];
    p2[1] = intersection_info[index * 3 + 2];
  }
}

void LineRenderer::render(
  Framebuffer &fb,
  Matrix<float32, 4, 4> transform,
  Array<Vec<float32,3>> starts,
  Array<Vec<float32,3>> ends,
  bool should_depth_be_zero)
{
  RasterBuffer raster(fb);
  DeviceRasterBuffer d_raster = raster.device_buffer();

  const int num_lines = starts.size();
  Vec<float32,3> *start_ptr =  starts.get_device_ptr();
  Vec<float32,3> *end_ptr =  ends.get_device_ptr();

  int32 width = fb.width();
  int32 height = fb.height();

  // save the colors and coordinates of the pixels to draw
  RAJA::forall<for_policy>(RAJA::RangeSegment(0, num_lines), [=] DRAY_LAMBDA (int32 i)
  {
    Vec<float32,4> color = {{0.f, 0.f, 0.f, 1.f}};

    Vec<float32,4> start;
    start[0] = start_ptr[i][0];
    start[1] = start_ptr[i][1];
    start[2] = start_ptr[i][2];
    start[3] = 1;

    Vec<float32,4> end;
    end[0] = end_ptr[i][0];
    end[1] = end_ptr[i][1];
    end[2] = end_ptr[i][2];
    end[3] = 1;

    start = transform * start;
    end = transform * end;

    float start_depth = start[3];
    float end_depth = end[3];

    // divide by the w component
    start = start / start[3];
    end = end / end[3];

    int x1,x2,y1,y2;
    x1 = ((start[0] + 1.f) / 2.f) * width;
    y1 = ((start[1] + 1.f) / 2.f) * height;
    x2 = ((end[0] + 1.f) / 2.f) * width;
    y2 = ((end[1] + 1.f) / 2.f) * height;

    Vec<int32, 2> p1, p2;
    p1[0] = x1;
    p1[1] = y1;
    p2[0] = x2;
    p2[1] = y2;
    crop_line_to_bounds(p1, p2, width, height);
    x1 = p1[0];
    y1 = p1[1];
    x2 = p2[0];
    y2 = p2[1];

    int myindex = 0;

    int dx = abs(x2 - x1);
    int sx = x1 < x2 ? 1 : -1;
    int dy = -1 * abs(y2 - y1);
    int sy = y1 < y2 ? 1 : -1;
    int err = dx + dy;

    int abs_dx = abs(dx);
    int abs_dy = abs(dy);
    float pixels_to_draw = 0.f;

    if (abs_dy > abs_dx)
    {
      // then slope is greater than 1
      // so we need one pixel for every y value
      pixels_to_draw = abs_dy + 1.f;
    }
    else
    {
      // then slope is less than 1
      // then we need one pixel for every x value
      pixels_to_draw = abs_dx + 1.f;
    }

    while (true)
    {
      float depth = 0.f;
      if (!should_depth_be_zero)
      {
        float progress = ((float) myindex) / pixels_to_draw;
        depth = (1.f - progress) * start_depth + progress * end_depth;
      }

      d_raster.write_pixel(x1, y1, color, depth);

      myindex += 1;
      if (x1 == x2 && y1 == y2)
      {
        break;
      }
      int e2 = 2 * err;
      if (e2 >= dy)
      {
        err += dy;
        x1 += sx;
      }
      if (e2 <= dx)
      {
        err += dx;
        y1 += sy;
      }
    }
  });

  // write this back to the original framebuffer
  raster.finalize();
}

void LineRenderer::render2(
  Framebuffer &fb,
  Matrix<float32, 4, 4> transform,
  Array<Vec<float32,3>> starts,
  Array<Vec<float32,3>> ends,
  bool should_depth_be_zero)
{
  RasterBuffer raster(fb);
  DeviceRasterBuffer d_raster = raster.device_buffer();

  const int num_lines = starts.size();
  Vec<float32,3> *start_ptr =  starts.get_device_ptr();
  Vec<float32,3> *end_ptr =  ends.get_device_ptr();

  Array<int32> pixels_per_line;
  pixels_per_line.resize(num_lines);
  int32 *pixels_per_line_ptr = pixels_per_line.get_device_ptr();

  int32 width = fb.width();
  int32 height = fb.height();

  Array<float32> start_depths;
  start_depths.resize(num_lines);
  float32 *start_depths_ptr = start_depths.get_device_ptr();
  Array<float32> end_depths;
  end_depths.resize(num_lines);
  float32 *end_depths_ptr = end_depths.get_device_ptr();

  Array<Vec<int32,4>> SS_starts_and_ends;
  SS_starts_and_ends.resize(num_lines);
  Vec<int32,4> *SS_starts_and_ends_ptr = SS_starts_and_ends.get_device_ptr();

  // count the number of pixels in each line
  RAJA::forall<for_policy>(RAJA::RangeSegment(0, num_lines), [=] DRAY_LAMBDA (int32 i)
  {
    Vec<float32,4> start;
    start[0] = start_ptr[i][0];
    start[1] = start_ptr[i][1];
    start[2] = start_ptr[i][2];
    start[3] = 1;

    Vec<float32,4> end;
    end[0] = end_ptr[i][0];
    end[1] = end_ptr[i][1];
    end[2] = end_ptr[i][2];
    end[3] = 1;

    start = transform * start;
    end = transform * end;

    start_depths_ptr[i] = start[3];
    end_depths_ptr[i] = end[3];

    // divide by the w component
    start = start / start[3];
    end = end / end[3];

    int x1,x2,y1,y2;
    x1 = ((start[0] + 1.f) / 2.f) * width;
    y1 = ((start[1] + 1.f) / 2.f) * height;
    x2 = ((end[0] + 1.f) / 2.f) * width;
    y2 = ((end[1] + 1.f) / 2.f) * height;

    Vec<int32, 2> p1, p2;
    p1[0] = x1;
    p1[1] = y1;
    p2[0] = x2;
    p2[1] = y2;
    crop_line_to_bounds(p1, p2, width, height);
    x1 = p1[0];
    y1 = p1[1];
    x2 = p2[0];
    y2 = p2[1];

    // we want to save these cropped and transformed values so we don't need to calculate them again
    SS_starts_and_ends_ptr[i] = {{x1,y1,x2,y2}};

    int dx = abs(x2 - x1);
    int dy = -1 * abs(y2 - y1);

    int abs_dx = abs(dx);
    int abs_dy = abs(dy);

    if (abs_dy > abs_dx)
    {
      // then slope is greater than 1
      // so we need one pixel for every y value
      pixels_per_line_ptr[i] = abs_dy + 1;
    }
    else
    {
      // then slope is less than 1
      // then we need one pixel for every x value
      pixels_per_line_ptr[i] = abs_dx + 1;
    }
  });

  // should this prefix sum be parallelized???
  // calculate offsets
  Array<int32> offsets;
  offsets.resize(num_lines);
  int32 *offsets_ptr = offsets.get_device_ptr();
  offsets_ptr[0] = 0;
  for (int32 i = 1; i < num_lines; i ++)
  {
    offsets_ptr[i] = offsets_ptr[i - 1] + pixels_per_line_ptr[i - 1];
  }

  int32 num_pixels = offsets_ptr[num_lines - 1] + pixels_per_line_ptr[num_lines - 1];

  // new containers for the next step's data
  Array<int32> x_values;
  Array<int32> y_values;
  Array<Vec<float32, 4>> colors;
  Array<float32> depths;
  x_values.resize(num_pixels);
  y_values.resize(num_pixels);
  colors.resize(num_pixels);
  depths.resize(num_pixels);
  int32 *x_values_ptr = x_values.get_device_ptr();
  int32 *y_values_ptr = y_values.get_device_ptr();
  Vec<float32, 4> *colors_ptr = colors.get_device_ptr();
  float32 *depths_ptr = depths.get_device_ptr();

  // save the colors and coordinates of the pixels to draw

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, num_lines), [=] DRAY_LAMBDA (int32 i)
  {
    Vec<float32,4> color = {{0.f, 0.f, 0.f, 1.f}};

    int32 x1,y1,x2,y2;
    x1 = SS_starts_and_ends_ptr[i][0];
    y1 = SS_starts_and_ends_ptr[i][1];
    x2 = SS_starts_and_ends_ptr[i][2];
    y2 = SS_starts_and_ends_ptr[i][3];

    int myindex = 0;

    int dx = abs(x2 - x1);
    int sx = x1 < x2 ? 1 : -1;
    int dy = -1 * abs(y2 - y1);
    int sy = y1 < y2 ? 1 : -1;
    int err = dx + dy;

    while (true)
    {
      float depth = 0.f;
      if (!should_depth_be_zero)
      {
        float progress = ((float32) myindex) / ((float32) pixels_per_line_ptr[i]);
        depth = (1.f - progress) * start_depths_ptr[i] + progress * end_depths_ptr[i];
      }

      x_values_ptr[myindex + offsets_ptr[i]] = x1;
      y_values_ptr[myindex + offsets_ptr[i]] = y1;
      colors_ptr[myindex + offsets_ptr[i]] = color;
      depths_ptr[myindex + offsets_ptr[i]] = depth;
      myindex += 1;
      if (x1 == x2 && y1 == y2)
      {
        break;
      }
      int e2 = 2 * err;
      if (e2 >= dy)
      {
        err += dy;
        x1 += sx;
      }
      if (e2 <= dx)
      {
        err += dx;
        y1 += sy;
      }
    }
  });

  // finally, render pixels
  RAJA::forall<for_policy>(RAJA::RangeSegment(0, num_pixels), [=] DRAY_LAMBDA (int32 i)
  {
    d_raster.write_pixel(x_values_ptr[i], y_values_ptr[i], colors_ptr[i], depths_ptr[i]);
  });

  // write this back to the original framebuffer
  raster.finalize();
}

void LineRenderer::render3(
  Framebuffer &fb,
  Matrix<float32, 4, 4> transform,
  Array<Vec<float32,3>> starts,
  Array<Vec<float32,3>> ends,
  bool should_depth_be_zero)
{
  RasterBuffer raster(fb);
  DeviceRasterBuffer d_raster = raster.device_buffer();

  const int num_lines = starts.size();
  Vec<float32,3> *start_ptr =  starts.get_device_ptr();
  Vec<float32,3> *end_ptr =  ends.get_device_ptr();

  int32 width = fb.width();
  int32 height = fb.height();

  // a new container for unit vectors along the lines
  Array<Vec<int32,2>> directions_array;
  directions_array.resize(num_lines);
  Vec<int32,2> *directions = directions_array.get_device_ptr();

  Array<int32> pixels_per_line;
  pixels_per_line.resize(num_lines);
  int32 *pixels_per_line_ptr = pixels_per_line.get_device_ptr();

  Array<float32> start_depths;
  start_depths.resize(num_lines);
  float32 *start_depths_ptr = start_depths.get_device_ptr();
  Array<float32> end_depths;
  end_depths.resize(num_lines);
  float32 *end_depths_ptr = end_depths.get_device_ptr();

  Array<Vec<int32,2>> SS_starts;
  SS_starts.resize(num_lines);
  Vec<int32,2> *SS_starts_ptr = SS_starts.get_device_ptr();

  // save the colors and coordinates of the pixels to draw
  RAJA::forall<for_policy>(RAJA::RangeSegment(0, num_lines), [=] DRAY_LAMBDA (int32 i)
  {
    Vec<float32,4> start;
    start[0] = start_ptr[i][0];
    start[1] = start_ptr[i][1];
    start[2] = start_ptr[i][2];
    start[3] = 1;

    Vec<float32,4> end;
    end[0] = end_ptr[i][0];
    end[1] = end_ptr[i][1];
    end[2] = end_ptr[i][2];
    end[3] = 1;

    start = transform * start;
    end = transform * end;

    start_depths_ptr[i] = start[3];
    end_depths_ptr[i] = end[3];

    // divide by the w component
    start = start / start[3];
    end = end / end[3];

    int x1,x2,y1,y2;
    x1 = ((start[0] + 1.f) / 2.f) * width;
    y1 = ((start[1] + 1.f) / 2.f) * height;
    x2 = ((end[0] + 1.f) / 2.f) * width;
    y2 = ((end[1] + 1.f) / 2.f) * height;

    Vec<int32, 2> p1, p2;
    p1[0] = x1;
    p1[1] = y1;
    p2[0] = x2;
    p2[1] = y2;
    crop_line_to_bounds(p1, p2, width, height);
    x1 = p1[0];
    y1 = p1[1];
    x2 = p2[0];
    y2 = p2[1];

    int dx = x2 - x1;
    int dy = y2 - y1;

    int abs_dx = abs(dx);
    int abs_dy = abs(dy);

    if (abs_dy > abs_dx)
    {
      // then slope is greater than 1
      // so we need one pixel for every y value
      pixels_per_line_ptr[i] = abs_dy + 1;
    }
    else
    {
      // then slope is less than 1
      // then we need one pixel for every x value
      pixels_per_line_ptr[i] = abs_dx + 1;
    }

    // we want to save these cropped and transformed values so we don't need to calculate them again
    SS_starts_ptr[i] = {{x1,y1}};

    directions[i] = {{dx, dy}};
  });

  // should this prefix sum be parallelized???
  // calculate offsets
  Array<int> offsets;
  offsets.resize(num_lines + 1);
  int *offsets_ptr = offsets.get_device_ptr();
  offsets_ptr[0] = 0;
  for (int i = 1; i < num_lines + 1; i ++)
  {
    offsets_ptr[i] = offsets_ptr[i - 1] + pixels_per_line_ptr[i - 1];
  }

  int num_pixels = offsets_ptr[num_lines];

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, num_pixels), [=] DRAY_LAMBDA (int32 i)
  {
    int which_line;
    int index;
    float percentage;

    for (int j = 0; j < num_lines; j ++)
    {
      if (offsets_ptr[j] <= i && offsets_ptr[j + 1] > i)
      {
        which_line = j;
        break;
      }
    }
    int32 offset = offsets_ptr[which_line];
    index = offset == 0 ? i : i % offset;

    percentage = ((float32) index) / ((float32) pixels_per_line_ptr[which_line]);
    float32 x1,y1;
    x1 = SS_starts_ptr[which_line][0];
    y1 = SS_starts_ptr[which_line][1];

    float32 dx, dy;
    dx = directions[which_line][0];
    dy = directions[which_line][1];

    int32 x,y;
    x = x1 + dx * percentage;
    y = y1 + dy * percentage;

    Vec<float32,4> color = {{0.f, 0.f, 0.f, 1.f}};

    float32 depth = 0.f;
    if (!should_depth_be_zero)
    {
      float32 progress = ((float32) index) / ((float32) pixels_per_line_ptr[which_line]);
      depth = (1.f - progress) * start_depths_ptr[which_line] + progress * end_depths_ptr[which_line];
    }

    d_raster.write_pixel(x, y, color, depth);
  });

  // write this back to the original framebuffer
  raster.finalize();
}

} // namespace dray

