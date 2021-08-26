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
#include <dray/rendering/font.hpp>
#include <dray/rendering/text_annotator.hpp>
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

  // next we will calculate intersections with the edges of the screen
  // and record if they are within bounds or not
  float32 x_intersect_bottom, y_intersect_bottom;
  bool bottom_intersect_ok = false;
  float32 x_intersect_top, y_intersect_top;
  bool top_intersect_ok = false;
  float32 x_intersect_left, y_intersect_left;
  bool left_intersect_ok = false;
  float32 x_intersect_right, y_intersect_right;
  bool right_intersect_ok = false;

  x_intersect_bottom = (-1 * b) / m;
  y_intersect_bottom = 0;
  if (x_intersect_bottom > -1 && x_intersect_bottom < width &&
    y_intersect_bottom > -1 && y_intersect_bottom < height)
  {
    bottom_intersect_ok = true;
  }

  x_intersect_top = ((height - 1) - b) / m;
  y_intersect_top = height - 1;
  if (x_intersect_top > -1 && x_intersect_top < width &&
    y_intersect_top > -1 && y_intersect_top < height)
  {
    top_intersect_ok = true;
  }

  x_intersect_left = 0;
  y_intersect_left = b;
  if (x_intersect_left > -1 && x_intersect_left < width &&
    y_intersect_left > -1 && y_intersect_left < height)
  {
    left_intersect_ok = true;
  }

  x_intersect_right = width - 1;
  y_intersect_right = m * (width - 1) + b;
  if (x_intersect_right > -1 && x_intersect_right < width &&
    y_intersect_right > -1 && y_intersect_right < height)
  {
    right_intersect_ok = true;
  }

  // tie breaking
  if (top_intersect_ok && bottom_intersect_ok)
  {
    left_intersect_ok = false;
    right_intersect_ok = false;
  }
  if (right_intersect_ok && left_intersect_ok)
  {
    top_intersect_ok = false;
    bottom_intersect_ok = false;
  }

  int top = 1;
  int bottom = 2;
  int left = 3;
  int right = 4;

  // next we set up a data structure to house information about our intersections
  // only two intersections will actually be in view of the camera
  // so we record which intersection it is, the distance^2 to p1, and the x and y vals
  Vec<float32, 8> info = {{0,0,0,0,0,0,0,0}};

  // either only two of the following conditions will ever be true
  // or only the first two that are true matter
  int index = 0;
  if (top_intersect_ok)
  {
    info[index + 0] = top;
    float y1_minus_newy = y1 - y_intersect_top;
    float x1_minus_newx = x1 - x_intersect_top;
    info[index + 1] = y1_minus_newy * y1_minus_newy + x1_minus_newx * x1_minus_newx;
    info[index + 2] = x_intersect_top;
    info[index + 3] = y_intersect_top;
    index += 4;
  }
  if (bottom_intersect_ok)
  {
    info[index + 0] = bottom;
    float y1_minus_newy = y1 - y_intersect_bottom;
    float x1_minus_newx = x1 - x_intersect_bottom;
    info[index + 1] = y1_minus_newy * y1_minus_newy + x1_minus_newx * x1_minus_newx;
    info[index + 2] = x_intersect_bottom;
    info[index + 3] = y_intersect_bottom;
    index += 4;
  }
  if (left_intersect_ok)
  {
    info[index + 0] = left;
    float y1_minus_newy = y1 - y_intersect_left;
    float x1_minus_newx = x1 - x_intersect_left;
    info[index + 1] = y1_minus_newy * y1_minus_newy + x1_minus_newx * x1_minus_newx;
    info[index + 2] = x_intersect_left;
    info[index + 3] = y_intersect_left;
    index += 4;
  }
  if (right_intersect_ok)
  {
    info[index + 0] = right;
    float y1_minus_newy = y1 - y_intersect_right;
    float x1_minus_newx = x1 - x_intersect_right;
    info[index + 1] = y1_minus_newy * y1_minus_newy + x1_minus_newx * x1_minus_newx;
    info[index + 2] = x_intersect_right;
    info[index + 3] = y_intersect_right;
  }

  // with this information we can assign new values to p1 and p2 if needed
  float32 distance1 = info[1];
  float32 distance2 = info[5];
  index = distance1 < distance2 ? 0 : 1;
  if (!p1_ok)
  {
    p1[0] = info[index * 4 + 2];
    p1[1] = info[index * 4 + 3];
  }

  index = !index;
  if (!p2_ok)
  {
    p2[0] = info[index * 4 + 2];
    p2[1] = info[index * 4 + 3];
  }
}

void LineRenderer::render_triad(
  Framebuffer &fb,
  Vec<int32, 2> pos,
  float32 distance,
  Camera &camera)
{
  RasterBuffer raster(fb);
  DeviceRasterBuffer d_raster = raster.device_buffer();

  int width = fb.width();
  int height = fb.height();

  Camera triad_camera;
  triad_camera.set_width (width);
  triad_camera.set_height (height);

  // set origin and basis vectors
  Vec<float32, 3> o = {{0,0,0}};
  Vec<float32, 3> i = {{1,0,0}};
  Vec<float32, 3> j = {{0,1,0}};
  Vec<float32, 3> k = {{0,0,1}};

  Vec<float32, 3> look = (camera.get_look_at() - camera.get_pos()).normalized();
  Vec<float32, 3> up = camera.get_up().normalized();

  triad_camera.set_pos(o - distance * look);
  triad_camera.set_up(up);
  triad_camera.set_look_at(o);

  Matrix<float32, 4, 4> V = triad_camera.view_matrix();

  o = transform_point(V, o);
  i = transform_point(V, i);
  j = transform_point(V, j);
  k = transform_point(V, k);

  int num_lines = 3;
  Array<Vec<float32,3>> starts;
  Array<Vec<float32,3>> ends;
  starts.resize(num_lines);
  ends.resize(num_lines);
  Vec<float32,3> *starts_ptr = starts.get_host_ptr();
  Vec<float32,3> *ends_ptr = ends.get_host_ptr();
  starts_ptr[0] = o;
  ends_ptr[0] = i;
  starts_ptr[1] = o;
  ends_ptr[1] = j;
  starts_ptr[2] = o;
  ends_ptr[2] = k;

  AABB<3> triad_aabb;
  triad_aabb.m_ranges[0].set_range(-1.f, 1.f);
  triad_aabb.m_ranges[1].set_range(-1.f, 1.f);
  triad_aabb.m_ranges[2].set_range(-1.f, 1.f);

  Matrix<float32, 4, 4> P = triad_camera.projection_matrix(triad_aabb);

  // for the triad labels
  Array<Vec<float32,2>> xyz_text_pos;
  xyz_text_pos.resize(3);
  Vec<float32,2> *xyz_text_pos_ptr = xyz_text_pos.get_host_ptr();

  // save the colors and coordinates of the pixels to draw
  RAJA::forall<for_policy>(RAJA::RangeSegment(0, num_lines), [=] DRAY_LAMBDA (int32 i)
  {
    Vec<float32,4> color = {{0.f, 0.f, 0.f, 1.f}};
    // float world_depth = 4.f;

    Vec<float32,4> start;
    start[0] = starts_ptr[i][0];
    start[1] = starts_ptr[i][1];
    start[2] = starts_ptr[i][2];
    start[3] = 1;

    Vec<float32,4> end;
    end[0] = ends_ptr[i][0];
    end[1] = ends_ptr[i][1];
    end[2] = ends_ptr[i][2];
    end[3] = 1;

    // for the annotations
    Vec<float32, 4> text_pos = P * ((end - start) * 1.1f + start);
    text_pos = text_pos / text_pos[3];
    int text_x,text_y;
    text_x = ((text_pos[0] + 1.f) / 2.f) * width;
    text_y = ((text_pos[1] + 1.f) / 2.f) * height;
    // depth is available here!!! just look at the z component


    start = P * start;
    end = P * end;

    // divide by the w component
    start = start / start[3];
    end = end / end[3];

    int x1,x2,y1,y2;
    x1 = ((start[0] + 1.f) / 2.f) * width;
    y1 = ((start[1] + 1.f) / 2.f) * height;
    x2 = ((end[0] + 1.f) / 2.f) * width;
    y2 = ((end[1] + 1.f) / 2.f) * height;

    // TODO crop these lines
    int xmov = pos[0] - x1;
    int ymov = pos[1] - y1;
    x1 = pos[0];
    y1 = pos[1];
    x2 += xmov;
    y2 += ymov;

    xyz_text_pos_ptr[i] = {{(float32) (text_x + xmov), (float32) (text_y + ymov)}};

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

      float progress = ((float) myindex) / pixels_to_draw;
      float depth = 0.1f;

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

  TextAnnotator annot;

  annot.add_text("X", xyz_text_pos_ptr[0], 20);
  annot.add_text("Y", xyz_text_pos_ptr[1], 20);
  annot.add_text("Z", xyz_text_pos_ptr[2], 20);

  annot.render(fb);
}

void LineRenderer::render(
  Framebuffer &fb,
  Matrix<float32, 4, 4> transform,
  Array<Vec<float32,3>> starts,
  Array<Vec<float32,3>> ends)
{
  RasterBuffer raster(fb);
  DeviceRasterBuffer d_raster = raster.device_buffer();

  const int num_lines = starts.size();
  Vec<float32,3> *start_ptr =  starts.get_device_ptr();
  Vec<float32,3> *end_ptr =  ends.get_device_ptr();

  float elapsed_time;
  Timer mytimer = Timer();
  mytimer.reset();

  int width = fb.width();
  int height = fb.height();

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
      // x_values_ptr[myindex + offsets_ptr[i]] = x1;
      // y_values_ptr[myindex + offsets_ptr[i]] = y1;
      // colors_ptr[myindex + offsets_ptr[i]] = color;
      // depths_ptr[myindex + offsets_ptr[i]] = world_depth;

      float progress = ((float) myindex) / pixels_to_draw;
      float depth = (1.f - progress) * start_depth + progress * end_depth;

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

  elapsed_time = mytimer.elapsed();
  std::cout << "elapsed time: " << elapsed_time << std::endl;

  // write this back to the original framebuffer
  raster.finalize();
}

void LineRenderer::justinrender(
  Framebuffer &fb,
  Matrix<float32, 4, 4> transform,
  Array<Vec<float32,3>> starts,
  Array<Vec<float32,3>> ends)
{
  RasterBuffer raster(fb);
  DeviceRasterBuffer d_raster = raster.device_buffer();

  const int num_lines = starts.size();
  Vec<float32,3> *start_ptr =  starts.get_device_ptr();
  Vec<float32,3> *end_ptr =  ends.get_device_ptr();

  float elapsed_time;
  Timer mytimer = Timer();
  mytimer.reset();

  // a new container for unit vectors along the lines
  Array<Vec<int32,2>> directions_array;
  directions_array.resize(num_lines);
  Vec<int32,2> *directions = directions_array.get_device_ptr();

  Array<int> pixels_per_line;
  pixels_per_line.resize(num_lines);
  int *pixels_per_line_ptr = pixels_per_line.get_device_ptr();

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

    int x1,x2,y1,y2;
    x1 = start[0];
    y1 = start[1];
    x2 = end[0];
    y2 = end[1];

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

    directions[i] = {{dx, dy}};
  });

  elapsed_time = mytimer.elapsed();
  std::cout << "lines loop elapsed time: " << elapsed_time << std::endl;
  mytimer.reset();

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

  elapsed_time = mytimer.elapsed();
  std::cout << "calc offsets elapsed time: " << elapsed_time << std::endl;
  mytimer.reset();

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
    int offset = offsets_ptr[which_line];
    index = offset == 0 ? i : i % offset;

    percentage = ((float) index) / ((float) pixels_per_line_ptr[which_line]);
    float x1,y1;
    x1 = start_ptr[which_line][0];
    y1 = start_ptr[which_line][1];

    float dx, dy;
    dx = directions[which_line][0];
    dy = directions[which_line][1];

    int x,y;
    x = x1 + dx * percentage;
    y = y1 + dy * percentage;

    Vec<float32,4> color = {{1.f, 0.f, 0.f, 1.f}};
    float world_depth = 4.f;
    d_raster.write_pixel(x, y, color, world_depth);
  });

  elapsed_time = mytimer.elapsed();
  std::cout << "pixels loop elapsed time: " << elapsed_time << std::endl;

  // write this back to the original framebuffer
  raster.finalize();
}

} // namespace dray

