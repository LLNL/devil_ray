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
  void get_aabb_lines_transformed_by_view(
    AABB<3> aabb,
    Array<Vec<float32,3>> &starts, 
    Array<Vec<float32,3>> &ends,
    Matrix<float32, 4, 4> V)
  {
    int num_lines = 12;
    starts.resize(num_lines);
    ends.resize(num_lines);
    Vec<float32,3> *starts_ptr = starts.get_host_ptr();
    Vec<float32,3> *ends_ptr = ends.get_host_ptr();

    float minx, miny, minz, maxx, maxy, maxz;
    minx = aabb.m_ranges[0].min();
    miny = aabb.m_ranges[1].min();
    minz = aabb.m_ranges[2].min();
    maxx = aabb.m_ranges[0].max();
    maxy = aabb.m_ranges[1].max();
    maxz = aabb.m_ranges[2].max();

    Vec<float32, 3> o,i,j,k,ij,ik,jk,ijk;
    o = transform_point(V, ((Vec<float32,3>) {{minx, miny, minz}}));
    i = transform_point(V, ((Vec<float32,3>) {{maxx, miny, minz}}));
    j = transform_point(V, ((Vec<float32,3>) {{minx, maxy, minz}}));
    k = transform_point(V, ((Vec<float32,3>) {{minx, miny, maxz}}));
    ij = transform_point(V, ((Vec<float32,3>) {{maxx, maxy, minz}}));
    ik = transform_point(V, ((Vec<float32,3>) {{maxx, miny, maxz}}));
    jk = transform_point(V, ((Vec<float32,3>) {{minx, maxy, maxz}}));
    ijk = transform_point(V, ((Vec<float32,3>) {{maxx, maxy, maxz}}));

    starts_ptr[0] = o;    ends_ptr[0] = i;
    starts_ptr[1] = o;    ends_ptr[1] = j;
    starts_ptr[2] = o;    ends_ptr[2] = k;
    starts_ptr[3] = i;    ends_ptr[3] = ik;
    starts_ptr[4] = i;    ends_ptr[4] = ij;
    starts_ptr[5] = j;    ends_ptr[5] = jk;
    starts_ptr[6] = j;    ends_ptr[6] = ij;
    starts_ptr[7] = ij;   ends_ptr[7] = ijk;
    starts_ptr[8] = k;    ends_ptr[8] = ik;
    starts_ptr[9] = k;    ends_ptr[9] = jk;
    starts_ptr[10] = ik;  ends_ptr[10] = ijk;
    starts_ptr[11] = jk;  ends_ptr[11] = ijk;
  }

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

} // namespace dray

