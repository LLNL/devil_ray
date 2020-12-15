// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_SAMPLING_HPP
#define DRAY_SAMPLING_HPP

#include <dray/types.hpp>
#include <dray/vec.hpp>
#include <dray/array.hpp>
#include <dray/error.hpp>


namespace dray
{

DRAY_EXEC
void create_basis(const Vec<Float, 3> &normal,
                   Vec<Float, 3> &xAxis,
                   Vec<Float, 3> &yAxis)
{
  // generate orthoganal basis about normal (i.e. basis for tangent space).
  // kz will be the axis idx (0,1,2) most aligned with normal.
  // TODO MAI [2018-05-30] I propose we instead choose the axis LEAST aligned with normal;
  // this amounts to flipping all the > to instead be <.
  int32 kz = 0;
  if (fabs (normal[0]) > fabs (normal[1]))
  {
    if (fabs (normal[0]) > fabs (normal[2]))
      kz = 0;
    else
      kz = 2;
  }
  else
  {
    if (fabs (normal[1]) > fabs (normal[2]))
      kz = 1;
    else
      kz = 2;
  }
  // nonNormal will be the axis vector most aligned with normal. (future: least aligned?)
  Vec<Float, 3> notNormal;
  notNormal[0] = 0.f;
  notNormal[1] = 0.f;
  notNormal[2] = 0.f;
  notNormal[(kz + 1) % 3] = 1.f; //[M.A.I. 5/31]

  xAxis = cross (normal, notNormal);
  xAxis.normalize ();
  yAxis = cross (normal, xAxis);
  yAxis.normalize ();
}

DRAY_EXEC
Vec<Float, 3>
cosine_weighted_hemisphere ( const Vec<float32,3> &normal,
                             const Vec<float32,2> &xy,
                             float32 &test_out)
{
  const float32 phi = 2.f * pi() * xy[0];
  const float32 cosTheta = sqrt(xy[1]), sinTheta = sqrt(1.0f - xy[1]);
  Vec<float32, 3> direction;
  direction[0] = cos(phi);
  direction[1] = sin(phi) * sinTheta;
  direction[2] = cosTheta;

  test_out = cosTheta;
  //const float32 r = sqrt (xy[0]);
  //const float32 theta = 2 * pi () * xy[1];

  //Vec<float32, 3> direction;
  //direction[0] = r * cos (theta);
  //direction[1] = r * sin (theta);
  //direction[2] = sqrt (max (0.0f, 1.f - xy[0]));

  // transform the direction into the normals orientation
  Vec<Float, 3> tangent_x, tangent_y;
  create_basis(normal, tangent_x, tangent_y);

  Vec<Float, 3> sample_dir = tangent_x * direction[0] +
                             tangent_y * direction[1] +
                             normal * direction[2];

  return sample_dir;
}

DRAY_EXEC
Vec<float32,3>
sphere_sample(const Vec<float32,3> &center,
              const float32 &radius,
              const Vec<float32,3> &hit_point,
              const Vec<float32,2> &u, // random
              float32 &pdf,
              bool debug = false)
{

  Vec<float32, 3> light_dir = center - hit_point;

  float32 dc = light_dir.magnitude();

  if(dc < radius)
  {
    std::cout<<"Inside sphere\n";
  }

  float32 invDc = 1.f / dc;
  Vec<float32, 3> wc = (center - hit_point) * invDc;
  Vec<float32,3> wcX, wcY;
  create_basis(wc,wcX,wcY);

  float32 sin_theta_max = radius * invDc;
  float32 sin_theta_max2 = sin_theta_max * sin_theta_max;
  float32 inv_sin_theta_max = 1 / sin_theta_max;
  float32 cos_theta_max = sqrt(max(0.f, 1.f - sin_theta_max2));

  float32 cos_theta = (cos_theta_max - 1.f) * u[0] + 1.f;
  float32 sin_theta2 = 1.f - cos_theta * cos_theta;

  float32 cos_alpha = sin_theta2 * inv_sin_theta_max + cos_theta *
    sqrt(max(0.f, 1.f - sin_theta2 * inv_sin_theta_max * inv_sin_theta_max));
  float32 sin_alpha = sqrt(max(0.f, 1.f - cos_alpha * cos_alpha));
  float32 phi = u[1] * 2.f * pi();


  // convert spherical coords, project to world
  // and get the point on the sphere. The coordinate
  // system was created to the sphere, so its reversed here
  Vec<float32,3> dir = sin_alpha * cos(phi) * (-wcX)
    + sin_alpha * sin(phi) * (-wcY) + cos_alpha * (-wc);
  Vec<float32,3> point = center + radius * dir;

  pdf = 1.f / (2.f * pi() * (1.f - cos_theta_max));
  if(debug) std::cout<<"cos theta max "<<cos_theta_max<<" pdf "<<pdf<<" \n";
  return point;
}

} // namespace dray
#endif
