// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_LOW_ORDER_INTERSECTORS_HPP
#define DRAY_LOW_ORDER_INTERSECTORS_HPP

#include <dray/types.hpp>
#include <dray/vec.hpp>
#include <dray/array.hpp>
#include <dray/error.hpp>


namespace dray
{

DRAY_EXEC
void quad_ref_point(const Vec<float32,3> &v00,
                    const Vec<float32,3> &v11,
                    const float32 &alpha,
                    const float32 &beta,
                    const Vec<float32,3> &e01,
                    const Vec<float32,3> &e03,
                    float32 &u,
                    float32 &v)
{

  constexpr float32 epsilon = 0.00001f;
  // Compute the barycentric coordinates of V11
  float32 alpha_11, beta_11;
  Vec<float32,3> e02 = v11 - v00;
  Vec<float32,3> n = cross(e01, e02);

  if ((abs(n[0]) >= abs(n[1])) && (abs(n[0]) >= abs(n[2])))
  {
    alpha_11 = ((e02[1] * e03[2]) - (e02[2] * e03[1])) / n[0];
    beta_11 = ((e01[1] * e02[2]) - (e01[2] * e02[1])) / n[0];
  }
  else if ((abs(n[1]) >= abs(n[0])) && (abs(n[1]) >= abs(n[2])))
  {
    alpha_11 = ((e02[2] * e03[0]) - (e02[0] * e03[2])) / n[1];
    beta_11 = ((e01[2] * e02[0]) - (e01[0] * e02[2])) / n[1];
  }
  else
  {
    alpha_11 = ((e02[0] * e03[1]) - (e02[1] * e03[0])) / n[2];
    beta_11 = ((e01[0] * e02[1]) - (e01[1] * e02[0])) / n[2];
  }

  // Compute the bilinear coordinates of the intersection point.
  if (abs(alpha_11 - 1.0f) < epsilon)
  {

    u = alpha;
    if (abs(beta_11 - 1.0f) < epsilon)
      v = beta;
    else
      v = beta / ((u * (beta_11 - 1.0f)) + 1.0f);
  }
  else if (abs(beta_11 - 1.0) < epsilon)
  {

    v = beta;
    u = alpha / ((v * (alpha_11 - 1.0f)) + 1.0f);
  }
  else
  {

    float32 a = 1.0f - beta_11;
    float32 b = (alpha * (beta_11 - 1.0f)) - (beta * (alpha_11 - 1.0f)) - 1.0f;
    float32 c = alpha;
    float32 d = (b * b) - (4.0f * a * c);
    float32 qq = -0.5f * (b + ((b < 0.0f ? -1.0f : 1.0f) * sqrt(d)));
    u = qq / a;
    if ((u < 0.0f) || (u > 1.0f))
    {
      u = c / qq;
    }
    v = beta / ((u * (beta_11 - 1.0f)) + 1.0f);
  }
}

DRAY_EXEC
float32 intersect_quad(const Vec<float32,3> &v00,
                       const Vec<float32,3> &v10,
                       const Vec<float32,3> &v11,
                       const Vec<float32,3> &v01,
                       const Vec<float32,3> &origin,
                       const Vec<float32,3> &dir,
                       float32 &alpha,
                       float32 &beta,
                       Vec<float32,3> &e01,
                       Vec<float32,3> &e03)
{
  constexpr float32 epsilon = 0.00001f;
  float32 distance = infinity32();
  /* An Eﬃcient Ray-Quadrilateral Intersection Test
     Ares Lagae Philip Dutr´e
     http://graphics.cs.kuleuven.be/publications/LD05ERQIT/index.html

  v01 *------------ * v11
      |\           |
      |  \         |
      |    \       |
      |      \     |
      |        \   |
      |          \ |
  v00 *------------* v10
  */
  // Rejects rays that are parallel to Q, and rays that intersect the plane of
  // Q either on the left of the line V00V01 or on the right of the line V00V10.

  e03 = v01 - v00;
  Vec<float32,3> p = cross(dir, e03);
  e01 = v10 - v00;
  float32 det = dot(e01, p);
  bool hit = true;

  if (abs(det) < epsilon)
  {
    hit = false;
  }
  float32 inv_det = 1.0f / det;
  Vec<float32,3> t = origin - v00;
  alpha = dot(t, p) * inv_det;
  if (alpha < 0.0)
  {
    hit = false;
  }
  Vec<float32,3> q = cross(t, e01);
  beta = dot(dir, q) * inv_det;
  if (beta < 0.0)
  {
    hit = false;
  }

  if ((alpha + beta) > 1.0f)
  {

    // Rejects rays that intersect the plane of Q either on the
    // left of the line V11V10 or on the right of the line V11V01.

    Vec<float32,3> e23 = v01 - v11;
    Vec<float32,3> e21 = v10 - v11;
    Vec<float32,3> p_prime = cross(dir, e21);
    float32 det_prime = dot(e23, p_prime);
    if (abs(det_prime) < epsilon)
    {
      hit = false;
    }
    float32 inv_det_prime = 1.0f / det_prime;
    Vec<float32,3> t_prime = origin - v11;
    float32 alpha_prime = dot(t_prime, p_prime) * inv_det_prime;
    if (alpha_prime < 0.0f)
    {
      hit = false;
    }
    Vec<float32,3> q_prime = cross(t_prime, e23);
    float32 beta_prime = dot(dir, q_prime) * inv_det_prime;
    if (beta_prime < 0.0f)
    {
      hit = false;
    }
  }

  // Compute the ray parameter of the intersection point, and
  // reject the ray if it does not hit Q.

  if(hit)
  {
    distance = dot(e03, q) * inv_det;
  }

  return distance;
}

DRAY_EXEC
float32 intersect_quad(const Vec<float32,3> &v00,
                       const Vec<float32,3> &v10,
                       const Vec<float32,3> &v01,
                       const Vec<float32,3> &v11,
                       const Vec<float32,3> &origin,
                       const Vec<float32,3> &dir,
                       float32 &u,
                       float32 &v)
{
  float32 alpha;
  float32 beta;
  Vec<float32,3> e01;
  Vec<float32,3> e03;
  float32 distance;
  distance = intersect_quad(v00, v10, v11, v01, origin, dir, alpha, beta, e01, e03);

  if(distance != infinity32())
  {
    quad_ref_point(v00, v11, alpha, beta, e01, e03, u, v);
  }

  return distance;
}

DRAY_EXEC
float32 intersect_quad(const Vec<float32,3> &v00,
                       const Vec<float32,3> &v10,
                       const Vec<float32,3> &v01,
                       const Vec<float32,3> &v11,
                       const Vec<float32,3> &origin,
                       const Vec<float32,3> &dir)
{
  float32 alpha;
  float32 beta;
  Vec<float32,3> e01;
  Vec<float32,3> e03;
  float32 distance;
  distance = intersect_quad(v00, v10, v11, v01, origin, dir, alpha, beta, e01, e03);

  return distance;
}


DRAY_EXEC
float32 intersect_tri(const Vec<float32,3> &a,
                      const Vec<float32,3> &b,
                      const Vec<float32,3> &c,
                      const Vec<float32,3> &origin,
                      const Vec<float32,3> &dir,
                      float32 &u,
                      float32 &v)
{
  const float32 EPSILON2 = 0.0001f;
  Float distance = infinity32();

  Vec<Float, 3> e1 = b - a;
  Vec<Float, 3> e2 = c - a;

  Vec<Float, 3> p;
  p[0] = dir[1] * e2[2] - dir[2] * e2[1];
  p[1] = dir[2] * e2[0] - dir[0] * e2[2];
  p[2] = dir[0] * e2[1] - dir[1] * e2[0];
  Float dot = e1[0] * p[0] + e1[1] * p[1] + e1[2] * p[2];
  if (dot != 0.f)
  {
    dot = 1.f / dot;
    Vec<Float, 3> t;
    t = origin - a;

    u = (t[0] * p[0] + t[1] * p[1] + t[2] * p[2]) * dot;
    if (u >= (0.f - EPSILON2) && u <= (1.f + EPSILON2))
    {

      Vec<Float, 3> q; // = t % e1;
      q[0] = t[1] * e1[2] - t[2] * e1[1];
      q[1] = t[2] * e1[0] - t[0] * e1[2];
      q[2] = t[0] * e1[1] - t[1] * e1[0];

      v = (dir[0] * q[0] +
           dir[1] * q[1] +
           dir[2] * q[2]) * dot;

      if (v >= (0.f - EPSILON2) && v <= (1.f + EPSILON2) && !(u + v > 1.f))
      {
        distance = (e2[0] * q[0] + e2[1] * q[1] + e2[2] * q[2]) * dot;
      }
    }
  }
  return distance;
}

DRAY_EXEC
float32 intersect_tri(const Vec<float32,3> &a,
                      const Vec<float32,3> &b,
                      const Vec<float32,3> &c,
                      const Vec<float32,3> &origin,
                      const Vec<float32,3> &dir)
{
  float32 u,v;
  float32 distance = intersect_tri(a,b,c,origin,dir,u,v);
  (void) u;
  (void) v;
  return distance;
}

DRAY_EXEC
float32 intersect_sphere(const Vec<float32,3> &center,
                         const float32 &radius,
                         const Vec<float32,3> &origin,
                         const Vec<float32,3> &dir)
{
  float32 dist = infinity32();

  Vec<float32, 3> l = center - origin;

  float32 dot1 = dot(l, dir);
  if (dot1 >= 0)
  {
    float32 d = dot(l, l) - dot1 * dot1;
    float32 r2 = radius * radius;
    if (d <= r2)
    {
      float32 tch = sqrt(r2 - d);
      dist = dot1 - tch;
    }
  }
  return dist;
}

} // namespace dray
#endif
