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
Vec3f reflect(const Vec3f &i, const Vec3f &n)
{
  return i - 2.f * dot(i, n) * n;
}

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
                             const Vec<float32,2> &xy)
{
  const float32 phi = 2.f * pi() * xy[0];
  const float32 cosTheta = sqrt(xy[1]);
  const float32 sinTheta = sqrt(1.0f - xy[1]);
  Vec<float32, 3> direction;
  direction[0] = cosTheta * cos(phi);
  direction[1] = cosTheta * sin(phi);;
  direction[2] = sinTheta;

  // transform the direction into the normals orientation
  Vec<Float, 3> tangent_x, tangent_y;
  create_basis(normal, tangent_x, tangent_y);

  Vec<Float, 3> sample_dir = tangent_x * direction[0] +
                             tangent_y * direction[1] +
                             normal * direction[2];

  return sample_dir;
}

DRAY_EXEC
Vec<Float, 3>
specular_sample( const Vec<float32,3> &normal,
                 const Vec<float32,3> &view,
                 const Vec<float32,2> &xy,
                 const float32 roughness,
                 bool debug = false)
{

  float32 phi = xy[0] * 2.f * pi();
  float32 r2 = roughness * roughness;

  float32 cos_theta = sqrt((1.f - xy[1]) / (1.f + (r2 - 1.f) * xy[1]));
  float32 sin_theta = clamp(sqrt(1.f - cos_theta * cos_theta),0.f,1.f);
  float32 sin_phi = sin(phi);
  float32 cos_phi = cos(phi);

  Vec<float32,3> half = {{sin_theta * cos_phi,
                          sin_theta * sin_phi,
                          cos_theta}};

  Vec<Float, 3> tangent_x, tangent_y;
  create_basis(normal, tangent_x, tangent_y);

  half = tangent_x * half[0] +
         tangent_y * half[1] +
         normal * half[2];

  Vec<float32,3> sample_dir = 2.0f * dot(view,half) * half - view;
  sample_dir.normalize();
  return sample_dir;
}

DRAY_EXEC
float32 schlick_fresnel(float32 u)
{
    float32 m = clamp(1.f - u, 0.f, 1.f);
    float32 m2 = m*m;
    return m2*m2*m; // pow(m,5)
}

DRAY_EXEC
float32 gtr2(float32 n_dot_h, float32 a)
{
  float32 a2 = a * a;
  float32 t = 1.0 + (a2 - 1.0) * n_dot_h * n_dot_h;
  return a2 / (pi() * t * t);
}

DRAY_EXEC
float32 smithg_ggx(float32 n_dot_v, float alpha_g)
{
    float32 a = alpha_g * alpha_g;
    float32 b = n_dot_v * n_dot_v;
    return 1.0f / (n_dot_v + sqrt(a + b - a * b));
}

DRAY_EXEC
Vec<float32,3>
eval_color(const Vec<float32,3> &normal,
           const Vec<float32,3> &sample_dir,
           const Vec<float32,3> &view,
           const Vec<float32,3> &base_color,
           const float32 roughness,
           const float32 diff_prob,
           bool debug = false)
{
  float32 n_dot_l = dot(normal,sample_dir);
  float32 n_dot_v = dot(normal,view);
  bool zero = false;
  // neither of these should be zero
  if(n_dot_l <= 0 || n_dot_v <= 0)
  {
    zero = true;
  }

  Vec<float32,3> h = sample_dir + view;
  h.normalize();
  float32 n_dot_h = dot(normal,h);
  float32 l_dot_h = dot(sample_dir,h);

  //https://github.com/wdas/brdf/blob/main/src/brdfs/disney.brdf
  // Diffuse fresnel - go from 1 at normal incidence to .5 at grazing
  // and mix in diffuse retro-reflection based on roughness
  //float32 fl = schlick_fresnel(n_dot_l);
  //float32 fv = schlick_fresnel(n_dot_v);
  //float fd90 = 0.5f + 2.f * l_dot_h * l_dot_h * roughness;
  //float fd = lerp(1.0f, fd90, fl) * lerp(1.0f, fd90, fv);

  //vec3 Cspec0 =
  // mix(specular*.08*mix(vec3(1), Ctint, specularTint), Cdlin, metallic);

  // simplified version of specular
  float32 min_val = 0.04;
  Vec<float32,3> spec_col;
  spec_col[0] = lerp(min_val, base_color[0], 1.f - diff_prob);
  spec_col[1] = lerp(min_val, base_color[1], 1.f - diff_prob);
  spec_col[2] = lerp(min_val, base_color[2], 1.f - diff_prob);
  float32 a = max(0.001f, roughness);
  float32 ds = gtr2(n_dot_h,a);

  // scale roughness into the range (.5, 1)
  a = 0.5f + a * 0.5f;
  float32 gs = smithg_ggx(n_dot_l, a)
               * smithg_ggx(n_dot_v, a);

  float32 fh = schlick_fresnel(l_dot_h);
  Vec<float32,3> fs;
  fs[0] = lerp(spec_col[0], 1.f, fh);
  fs[1] = lerp(spec_col[1], 1.f, fh);
  fs[2] = lerp(spec_col[2], 1.f, fh);

  Vec<float32,3> sample_color = (base_color / pi()) * diff_prob + gs * fs * ds;
  if(zero)
  {
    sample_color = {{0.f,0.f,0.f}};
  }

  if(debug)
  {
    std::cout<<"[Sample eval] sample color "<<sample_color<<"\n";
    std::cout<<"[Sample eval] spec_col "<<spec_col<<"\n";
    std::cout<<"[Sample eval] half "<<h<<"\n";
    std::cout<<"[Sample eval] gs "<<gs<<"\n";
    std::cout<<"[Sample eval] fs "<<fs<<"\n";
    std::cout<<"[Sample eval] ds "<<ds<<"\n";
    std::cout<<"[Sample eval] n_dot_h "<<n_dot_h<<"\n";
    std::cout<<"[Sample eval] n_dot_l "<<n_dot_l<<"\n";
    //std::cout<<"[Sample eval] fd "<<fd<<"\n";
    std::cout<<"[Sample eval] smith1 "<<smithg_ggx(n_dot_l, a)<<"\n";
    std::cout<<"[Sample eval] smith2 "<<smithg_ggx(n_dot_v, a)<<"\n";
  }
  return sample_color;
}


DRAY_EXEC
float32
eval_pdf(const Vec<float32,3> &sample_dir,
         const Vec<float32,3> &view,
         const Vec<float32,3> &normal,
         const float32 roughness,
         const float32 diff_prob,
         bool debug = false)
{
  // evaluate the pdf
  Vec<float32,3> h = sample_dir + view;
  h.normalize();
  float32 n_dot_h = dot(normal,h);
  float32 l_dot_h = dot(sample_dir,h);
  float32 spec_prob = 1.f - diff_prob;
  float32 cos_theta = abs(n_dot_h);
  float32 gtr2_pdf = gtr2(cos_theta, roughness) * cos_theta;

  float32 pdf_spec = gtr2_pdf / (4.f * abs(l_dot_h) +0.05f);
  float32 pdf_diff = abs(dot(normal,sample_dir)) * (1.0 / pi());
  float32 pdf = pdf_spec * spec_prob + diff_prob * pdf_diff;

  if(debug)
  {
    std::cout<<"[Sample pdf]  mix diff "<<diff_prob<<" spec "<<spec_prob<<"\n";
    std::cout<<"[Sample pdf]  l_dot_h "<<l_dot_h<<"\n";
    std::cout<<"[Sample pdf]  gtr2 "<<gtr2_pdf<<"\n";
    std::cout<<"[Sample pdf]  spec "<<pdf_spec<<"\n";
    std::cout<<"[Sample pdf]  diff "<<pdf_diff<<"\n";
    std::cout<<"[Sample pdf]  cos_theta "<<cos_theta<<" pdf "<<pdf<<"\n";
  }

  return pdf;
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
