// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_DISNEY_SAMPLING_HPP
#define DRAY_DISNEY_SAMPLING_HPP

#include <dray/rendering/sampling.hpp>
#include <dray/rendering/path_data.hpp>
#include <dray/random.hpp>
#include <dray/matrix.hpp>

namespace dray
{


// sampling convientions
// wo = tangent space direction of output direction.
//      this is the view direction or -ray.m_dir
// wi = tangent space direction of input direction
//      this is ths sample direction (incoming light dir)
// wh = tangent space direction of the half vector
//      (wo + wi).normalize


// trig functions for normalized vectors in tangent space
// where the normal is (0,0,1)
DRAY_EXEC float32 tcos_theta(const Vec<float32,3> &dir)
{
  return dir[2];
}

DRAY_EXEC float32 tcos2_theta(const Vec<float32,3> &dir)
{
  return dir[2] * dir[2];
}

DRAY_EXEC float32 tsin2_theta(const Vec<float32,3> &dir)
{
  return max(0.f, 1.f - tcos2_theta(dir));
}

DRAY_EXEC float32 tsin_theta(const Vec<float32,3> &dir)
{
  return sqrt(tsin2_theta(dir));
}

DRAY_EXEC float32 ttan_theta(const Vec<float32,3> &dir)
{
  return tsin_theta(dir) / tcos_theta(dir);
}

DRAY_EXEC float32 ttan2_theta(const Vec<float32,3> &dir)
{
  return tsin2_theta(dir) / tcos2_theta(dir);
}

DRAY_EXEC float32 tcos_phi(const Vec<float32,3> &dir)
{
  float32 sin_theta = tsin_theta(dir);
  return sin_theta == 0.f ? 1.f : clamp(dir[0] / sin_theta, -1.f, 1.f);
}

DRAY_EXEC float32 tsin_phi(const Vec<float32,3> &dir)
{
  float32 sin_theta = tsin_theta(dir);
  return sin_theta == 0.f ? 0.f : clamp(dir[1] / sin_theta, -1.f, 1.f);
}

DRAY_EXEC float32 tsin2_phi(const Vec<float32,3> &dir)
{
  return tsin_phi(dir) * tsin_phi(dir);
}

DRAY_EXEC float32 tcos2_phi(const Vec<float32,3> &dir)
{
  return tcos_phi(dir) * tcos_phi(dir);
}

DRAY_EXEC
void calc_anisotropic(float32 roughness, float32 anisotropic, float32 &ax, float32 &ay)
{
  float32 aspect = sqrtf(1.0f - 0.9f * anisotropic);
  ax = max(0.001f, sqrt(roughness) / aspect);
  ay = max(0.001f, sqrt(roughness) * aspect);
}

DRAY_EXEC
float32 smithg_ggx_aniso(float32 n_dot_v,
                         float32 v_dot_x,
                         float32 v_dot_y,
                         float32 ax,
                         float32 ay)
{
  float32 a = v_dot_x * ax;
  float32 b = v_dot_y * ay;
  float32 c = n_dot_v;
  return 1.0 / (n_dot_v + sqrt(a*a + b*b + c*c));
}

DRAY_EXEC
float32 separable_ggx_aniso(const Vec<float32,3> &w,
                            float32 ax,
                            float32 ay)
{
  // just do this calculation in tangent space
  float32 cos_theta = tcos_theta(w);
  if(cos_theta == 0.f)
  {
    std::cout<<"cos theta 0 "<<w<<"\n";
    return 0.f;
  }

  float32 abs_tan_theta = abs(ttan_theta(w));
  float32 a = sqrt(tcos2_phi(w) * ax * ax + tsin2_phi(w) * ay * ay);
  float32 b = a * a * abs_tan_theta * abs_tan_theta;
  float32 lambda = 0.5f * (-1.f + sqrt(1.f * b));

  return 1.f / (1.f + lambda);
}

DRAY_EXEC
float32 gtr1(float32 n_dot_h, float32 a)
{
  if (a >= 1.0)
      return (1.0 / pi());
  float32 a2 = a * a;
  float32 t = 1.0 + (a2 - 1.0) * n_dot_h * n_dot_h;
  return (a2 - 1.0) / (pi() * log(a2) * t);
}

DRAY_EXEC
float32 fresnel(float32 theta, float32 n1, float32 n2)
{
  float32 r0 = (n1 - n2) / (n1 + n2);
  r0 = r0 * r0;
  return r0 + (1.f - r0) * schlick_fresnel(theta);
}

//http://www.jcgt.org/published/0007/04/01/paper.pdf
// GgxAnisotropicD
DRAY_EXEC
float32 gtr2_aniso(const Vec<float32,3> &wh,
                   float32 ax,
                   float32 ay, bool debug = false)
{
  float32 x2 = wh[0] * wh[0];
  float32 y2 = wh[1] * wh[1];
  float32 z2 = wh[2] * wh[2]; // cos2theta
  float32 ax2 = ax * ax;
  float32 ay2 = ay * ay;
  float32 c = x2 / ax2 + y2 / ay2 + z2;
  float32 c2 = c * c;

  float32 d = ax * ay * c2;
  if(debug)
  {
    std::cout<<"[gtr2] wh "<<wh<<"\n";
    std::cout<<"[gtr2] ax "<<ax<<"\n";
    std::cout<<"[gtr2] ay "<<ax<<"\n";
    std::cout<<"[gtr2] c c2 d "<<c<<" "<<c2<<" "<<d<<"\n";
  }
  return 1.f / (pi() * d);
  //float32 h_dot_x = wh[0];
  //float32 h_dot_y = wh[1];
  //float32 n_dot_h = tcos_theta(wh);

  //float32 a = h_dot_x / ax;
  //float32 b = h_dot_y / ay;
  //float32 c = a * a + b * b + n_dot_h * n_dot_h;
  //if(debug)
  //{
  //  std::cout<<"[gtr2] a b c "<<a<<" "<<b<<" "<<c<<"\n";
  //}

  //return 1.0f / (pi() * ax * ay * c * c);
}

DRAY_EXEC
float32 dielectric(float32 cosThetaI, float32 ni, float32 nt)
{
    // Copied from PBRT. This function calculates the
    // full Fresnel term for a dielectric material.

    cosThetaI = clamp(cosThetaI, -1.0f, 1.0f);

    // Swap index of refraction if this is coming from inside the surface
    if(cosThetaI < 0.0f)
    {
      float temp = ni;
      ni = nt;
      nt = temp;
      cosThetaI = -cosThetaI;
    }

    float sinThetaI = sqrtf(max(0.0f, 1.0f - cosThetaI * cosThetaI));
    float sinThetaT = ni / nt * sinThetaI;

    // Check for total internal reflection
    if(sinThetaT >= 1) {
        return 1;
    }

    float cosThetaT = sqrtf(max(0.0f, 1.0f - sinThetaT * sinThetaT));

    float rParallel = ((nt * cosThetaI) - (ni * cosThetaT)) / ((nt * cosThetaI) + (ni * cosThetaT));
    float rPerpendicuar = ((ni * cosThetaI) - (nt * cosThetaT)) / ((ni * cosThetaI) + (nt * cosThetaT));
    return (rParallel * rParallel + rPerpendicuar * rPerpendicuar) / 2;
}

DRAY_EXEC
Vec<float32,3> sample_ggx(float32 roughness, Vec<float32,2> rand)
{
  float32 a = max(0.001f, roughness);

  float32 phi = rand[0] * 2.f * pi();

  float32 cos_theta = sqrt((1.f - rand[1]) / (1.f + (a * a - 1.f) * rand[1]));
  float32 sin_theta = clamp(sqrt(1.f - (cos_theta * cos_theta)), 0.f, 1.f);
  float32 sin_phi = sin(phi);
  float32 cos_phi = cos(phi);

  Vec<float32,3> dir;
  dir[0] = sin_theta * cos_phi;
  dir[1] = sin_theta * sin_phi;
  dir[2] = cos_theta;

  dir.normalize();
  return dir;
}

DRAY_EXEC
Vec<float32,3> sample_vndf_ggx(const Vec<float32,3> &wo,
                               float32 ax,
                               float32 ay,
                               Vec<float32,2> rand)
{

  // this code samples based on the assumption that
  // the view direction is in tangent space
  // https://hal.archives-ouvertes.fr/hal-01509746/document

  // stretched view vector
  Vec<float32,3> s_view;
  s_view[0] = wo[0] * ax;
  s_view[1] = wo[1] * ay;
  s_view[2] = wo[2];
  s_view.normalize();

  Vec<float32,3> wcX, wcY;
  create_basis(s_view,wcX,wcY);

  float32 a = 1.f / (1.f + s_view[2]);
  float32 r = sqrt(rand[0]);

  float32 phi;
  if(rand[1] < a)
  {
    phi = rand[1] / a * pi();
  }
  else
  {
    phi = pi() + (rand[1] - a) / (1.f - a) * pi();
  }
  float32 p1 = r * cos(phi);
  float32 p2 = r * sin(phi) * (( rand[1] < a) ? 1.f : s_view[2]);

  // dir is the half vector
  Vec<float32,3> dir;
  float32 p3 = sqrt(max(0.f, 1.f - p1 * p1 - p2 * p2));

  dir = p1 * wcX + p2 * wcY + p3 * s_view;
  dir[0] *= ax * s_view[0];
  dir[1] *= ay * s_view[1];
  dir[2] *= s_view[2];
  dir.normalize();

  return dir;
}

DRAY_EXEC
float32 pdf_vndf_ggx(const Vec<float32,3> &wo,
                     const Vec<float32,3> &wi,
                     const float32 ax,
                     const float32 ay,
                     bool debug = false)
{

  Vec<float32,3> wh = wo + wi;
  wh.normalize();

  // http://www.jcgt.org/published/0007/04/01/paper.pdf
  // eq 3
  // distribution of visible normals
  // Dv(Ni) = G1(V) * max(0, dot(N,V) / dot(V,Z)
  // where Z = (0,0,1), i.e., dot(V,Z) = costheta(V)
  // N = half angle
  float32 g = separable_ggx_aniso(wo, ax, ay);

  float32 d = gtr2_aniso(wh, ax, ay, debug);

  float32 abs_n_dot_l = abs(tcos_theta(wi));
  float32 abs_h_dot_l = abs(dot(wh,wi));
  if(debug)
  {
    std::cout<<"[ VNDF pdf ] g "<<g<<"\n";
    std::cout<<"[ VNDF pdf ] d "<<g<<"\n";
  }

  return g * abs_h_dot_l * d / abs_n_dot_l;
}



DRAY_EXEC
float32 disney_pdf(const Vec<float32,3> &wo,
                   const Vec<float32,3> &wi,
                   const Material &mat,
                   bool debug = false)
{
  Vec<float32,3> wh = wo + wi;
  wh.normalize();

  float32 n_dot_h = tcos_theta(wh);

  if(debug)
  {
    std::cout<<"[PDF] n_dot_l "<<tcos_theta(wi)<<"\n";
  }

  if(tcos_theta(wi) < 0)
  {
    // Since we are modeling thin refraction
    // I think this should be the specular pdf
    // using the -half angle.
    // maybe mixing the odds of reflection versus
    // refraction
    return 1.f;
  }

  float32 specular_alpha = max(0.001f, mat.m_roughness);

  float32 diff_prob = 1.f - mat.m_metallic;
  float32 spec_prob = mat.m_metallic;

  float32 ax,ay;
  calc_anisotropic(mat.m_roughness, mat.m_anisotropic, ax, ay);

  // visible normal importance sampling pdf
  float32 vndf_pdf = pdf_vndf_ggx(wo, wi, ax, ay);

  // clearcloat pdf
  float32 clearcoat_alpha = mix(0.1f,0.001f, mat.m_clearcoat_gloss);
  float32 clearcoat_pdf =  gtr1(n_dot_h, clearcoat_alpha) * n_dot_h;
  float32 mix_ratio = 1.f / (1.f + mat.m_clearcoat);

  float32 pdf_spec = mix(clearcoat_pdf, vndf_pdf, mix_ratio) / (4.f * abs(dot(wh,wi)));

  // diffuse pdf
  float32 pdf_diff = tcos_theta(wi) / pi();

  // total brdf pdf
  float32 brdf_pdf = diff_prob * pdf_diff + spec_prob * pdf_spec;

  //
  float32 thin_roughness = mat.m_roughness * clamp(0.65f * mat.m_ior - 0.35f, 0.f, 1.f);
  float32 h_dot_v = dot(wh, wo);
  float32 f = dielectric(h_dot_v, 1.0f, mat.m_ior);
  float32 reflect_pdf = 1.f / (4.f * abs(dot(wo,wh)));
  float32 l_dot_h = dot(wi,wh);
  float32 den = l_dot_h + mat.m_ior * dot(wo,wh);
  float32 refract_pdf = l_dot_h / (den * den);
  float32 scaled_vndf_pdf = pdf_vndf_ggx(wo, wi, ax * thin_roughness, ay * thin_roughness);
  float32 bsdf_pdf = scaled_vndf_pdf * mix(refract_pdf, reflect_pdf, f);
  //float32 g2_iso = gtr2(n_dot_h, specular_alpha) * n_dot_h;
  //float32 fres = fresnel(abs(dot(sample_dir,half)), 1.f, mat.m_ior);
  //float32 bsdf_pdf = g2_iso * fres / (4.f * dot(sample_dir,half));

  float32 pdf = mix(brdf_pdf,bsdf_pdf, mat.m_spec_trans);

  if(debug)
  {
    std::cout<<"[PDF pdf_spec] "<<pdf_spec<<"\n";
    std::cout<<"[PDF pdf_diff] "<<pdf_diff<<"\n";
    std::cout<<"[PDF pdf_brdf] "<<brdf_pdf<<"\n";
    std::cout<<"[PDF pdf_bsdf] "<<bsdf_pdf<<"\n";
    std::cout<<"[PDF pdf] "<<pdf<<"\n";
  }
  return pdf;
}

DRAY_EXEC
Vec<float32,3> sample_spec_trans(const Vec<float32,3> &wo,
                                 const Material &mat,
                                 bool &specular,
                                 Vec<uint,2> &rand_state,
                                 bool debug = false)
{
  Vec<float32,3> wi;
  float32 ax,ay;
  calc_anisotropic(mat.m_roughness, mat.m_anisotropic, ax, ay);
  // always use air
  float32 n_air = 1.0;
  float32 n_mat = mat.m_ior;
  Vec<float32,2> rand;
  rand[0] = randomf(rand_state);
  rand[1] = randomf(rand_state);
  float32 thin_roughness = mat.m_roughness * clamp(0.65f * mat.m_ior - 0.35f, 0.f, 1.f);

  Vec<float32,3> wh = sample_vndf_ggx(wo, ax * thin_roughness, ay * thin_roughness, rand);

  if(debug)
  {
    std::cout<<"[Sample] wo "<<wo<<"\n";
    std::cout<<"[Sample] wh "<<wh<<"\n";
  }

  float32 v_dot_h = dot(wo,wh);
  if(wh[2] < 0.f)
  {
    v_dot_h = -v_dot_h;
  }

  float32 f = dielectric(v_dot_h, 1.0f, mat.m_ior);

  if(debug)
  {
    std::cout<<"[Sample] transmission\n";
  }

  if(randomf(rand_state) < f)
  {
    wi = reflect(wh,wo);
    wi[2] = -wi[2];
    if(debug)
    {
      std::cout<<"[Sample] refect\n";
    }
  }
  else
  {
    // normally we would refract, but we are only modeling
    // thin surfaces, so reflect and flip
    wi = reflect(wh,wo);
    //wi[2] = -wi[2];
    specular = true;
    if(debug)
    {
      std::cout<<"[Sample] refract\n";
      std::cout<<"[Sample] dot v_dot_h "<<dot(wh,wo)<<"\n";
      std::cout<<"[Sample] dot l_dot_h "<<dot(wi,wo)<<"\n";
    }
  }
  wi.normalize();
  return wi;
}

float32 pdf_spec_trans(Vec<float32,3> wi,
                       const Vec<float32,3> &wo,
                       const Material &mat,
                       bool debug = false)
{

  // we are modeling thin transmission so we can't scale the half angle
  // by the index of refraction to correct the half angle.
  // The half angle was sampled in the upper hemi, so if this
  // was refraction (lower hemi) then we have to flip it back up
  // to evaluate it correctly
  bool reflection = true;
  if(tcos_theta(wi) < 0.f)
  {
    wi[2] = -wi[2];
    reflection = false;
  }

  Vec<float32,3> wh = wi + wo;
  wh.normalize();


  float32 ax,ay;
  calc_anisotropic(mat.m_roughness, mat.m_anisotropic, ax, ay);

  float32 thin_roughness = mat.m_roughness * clamp(0.65f * mat.m_ior - 0.35f, 0.f, 1.f);
  // distribution pdf
  float32 distribution_pdf = pdf_vndf_ggx(wo, wi, ax * thin_roughness, ay * thin_roughness, debug);

  float32 h_dot_v = dot(wh, wo);
  float32 f = dielectric(h_dot_v, 1.0f, mat.m_ior);
  float32 reflect_pdf = f * distribution_pdf / (4.f * abs(dot(wo,wh)));
  if(!reflection)
  {
    // it can only be one or the other
    reflect_pdf = 0;
  }

  float32 l_dot_h = dot(wi,wh);
  float32 den = l_dot_h + mat.m_ior * dot(wo,wh);
  float32 refract_pdf = l_dot_h / (den * den);
  refract_pdf *= (1.f - f) * distribution_pdf;
  if(reflection)
  {
    // it can only be one or the other
    refract_pdf = 0;
  }

  if(debug)
  {
    std::cout<<"[PDF spec trans] reflect "<<reflect_pdf<<"\n";
    std::cout<<"[PDF spec trans] refract "<<refract_pdf<<"\n";
    std::cout<<"[PDF spec trans] distribution pdf "<<distribution_pdf<<"\n";
    std::cout<<"[PDF spec trans] frensel "<<f<<"\n";
  }

  float32 bsdf_pdf = refract_pdf + reflect_pdf;
  return bsdf_pdf;
}

DRAY_EXEC
Vec<float32,3> eval_spec_trans(const Vec<float32,3> &base_color,
                               Vec<float32,3> wi,
                               const Vec<float32,3> &wo,
                               const Material &mat,
                               bool debug = false)
{
  Vec<float32,3> bsdf = {{0.f, 0.f, 0.f}};

  if(debug)
  {
    std::cout<<"[Color eval] base_color "<<base_color<<"\n";
  }

  // we are modeling thin transmission so we can't scale the half angle
  // by the index of refraction to correct the half angle.
  // The half angle was sampled in the upper hemi, so if this
  // was refraction (lower hemi) then we have to flip it back up
  // to evaluate it correctly
  if(tcos_theta(wi) < 0.f)
  {
    wi[2] = -wi[2];
  }
  else
  {
    // if the sample is in the upper hemi, then there was
    // no transmission
    return bsdf;
  }

  float32 n_dot_l = tcos_theta(wi);
  float32 n_dot_v = tcos_theta(wo);

  float32 ax,ay;
  calc_anisotropic(mat.m_roughness, mat.m_anisotropic, ax, ay);
  float32 thin_roughness = mat.m_roughness * clamp(0.65f * mat.m_ior - 0.35f, 0.f, 1.f);
  ax *= thin_roughness;
  ay *= thin_roughness;

  Vec<float32,3> wh = wi + wo;
  wh.normalize();
  wh[2] = -wh[2];

  if(debug)
  {
    std::cout<<"[Color eval] wi "<<wi<<"\n";
    std::cout<<"[Color eval] wo "<<wo<<"\n";
    std::cout<<"[Color eval] wh "<<wh<<"\n";
  }

  float32 n_dot_h = tcos_theta(wh);
  float32 l_dot_h = dot(wi, wh);

  //float32 ni = wo[2] > 0.0f ? 1.0f : mat.m_ior;
  //float32 nt = wo[2] > 0.0f ? mat.m_ior : 1.0f;
  //float32 eta = ni / nt;

  //Vec<float32,3> twh = (wo + eta * wi);
  //twh.normalize();

  if(debug)
  {
     std::cout<<"[Color eval] thin rough "<<thin_roughness<<"\n";
  }

  // thin transmission
  bsdf[0] = sqrt(base_color[0]);
  bsdf[1] = sqrt(base_color[1]);
  bsdf[2] = sqrt(base_color[2]);

  //bsdf[0] = base_color[0];
  //bsdf[1] = base_color[1];
  //bsdf[2] = base_color[2];
  if(debug)
  {
    std::cout<<"[Color eval] bsdf sqrt "<<bsdf<<"\n";
    std::cout<<"[Color eval] bsdf ax ay "<<ax<<" "<<ay<<"\n";
    std::cout<<"[Color eval] bsdf wh "<<wh<<"\n";
  }


  // microfacet brdf
  float32 d = gtr2_aniso(wh, ax, ay, debug);
  float32 gl = separable_ggx_aniso(wi, ax, ay);
  float32 gv = separable_ggx_aniso(wo, ax, ay);
  float32 f = dielectric(dot(wo,wh), 1.0f, mat.m_ior);

  float32 c = (abs(dot(wi,wh)) * abs(dot(wh,wo))) / (abs(n_dot_l) * abs(n_dot_v));
  float32 n2 = mat.m_ior * mat.m_ior;
  float32 den = l_dot_h + mat.m_ior * dot(wh,wo);
  float32 t = n2 / ( den * den);

  bsdf = bsdf * c * t * (1.f - f) * gl * gv * d;

  if(debug)
  {
    std::cout<<"[Color eval] bsdf color "<<bsdf<<"\n";
    std::cout<<"[Color eval] bsdf masking gl "<<gl<<"\n";
    std::cout<<"[Color eval] bsdf masking gv "<<gv<<"\n";
    std::cout<<"[Color eval] bsdf t "<<t<<"\n";
    std::cout<<"[Color eval] bsdf den "<<den<<"\n";
    std::cout<<"[Color eval] bsdf d "<<d<<"\n";
    std::cout<<"[Color eval] bsdf c "<<c<<"\n";
    std::cout<<"[Color eval] bsdf f "<<f<<"\n";
    std::cout<<"[Color eval] bsdf abs_l_dot_h  "<<abs(l_dot_h)<<"\n";
    std::cout<<"[Color eval] bsdf abs_v_dot_h  "<<abs(dot(wh,wo))<<"\n";
    std::cout<<"[Color eval] bsdf abs_n_dot_l  "<<abs(n_dot_l)<<"\n";
    std::cout<<"[Color eval] bsdf abs_n_dot_v  "<<abs(n_dot_v)<<"\n";
    std::cout<<"[Color eval] bsdf h_dot_n "<<tcos_theta(wh)<<"\n";
  }
  return bsdf;
}

DRAY_EXEC
Vec<float32,3> sample_disney(const Vec<float32,3> &wo,
                             const Material &mat,
                             bool &specular,
                             Vec<uint,2> &rand_state,
                             bool debug = false)
{

  Vec<float32,3> wi;
  float32 ax,ay;
  calc_anisotropic(mat.m_roughness, mat.m_anisotropic, ax, ay);

  float32 spec_trans_roll = randomf(rand_state);
  if(debug)
  {
    std::cout<<"[Sample] spec_trans roll "<<spec_trans_roll<<"\n";
    std::cout<<"[Sample] spec_trans "<<mat.m_spec_trans<<"\n";
  }

  if(mat.m_spec_trans > spec_trans_roll)
  {
    wi = sample_spec_trans(wo, mat, specular, rand_state, debug);
  }
  else
  {
    float32 diff_prob = 1.f - mat.m_metallic;
    Vec<float32,2> rand;
    rand[0] = randomf(rand_state);
    rand[1] = randomf(rand_state);
    if(randomf(rand_state) < diff_prob)
    {
      wi = cosine_weighted_hemisphere(rand);
      specular = false;
      if(debug)
      {
        std::cout<<"[Sample] diffuse\n";
        std::cout<<"[Sample] n_dot_l "<<tcos_theta(wi)<<"\n";
      }
    }
    else
    {
      Vec<float32,3> wh = sample_vndf_ggx(wo, ax, ay, rand);
      wi = reflect(wh,wo);
      specular = true;
      if(debug)
      {
        std::cout<<"[Sample] specular\n";
      }
    }

  }

  wi.normalize();
  return wi;
}



DRAY_EXEC
Vec<float32,3> eval_disney(const Vec<float32,3> &base_color,
                           const Vec<float32,3> &wi,
                           const Vec<float32,3> &wo,
                           const Material &mat,
                           bool debug = false)
{
  Vec<float32,3> color = {{0.f, 0.f, 0.f}};
  Vec<float32,3> brdf = {{0.f, 0.f, 0.f}};
  Vec<float32,3> bsdf = {{0.f, 0.f, 0.f}};
  if(debug)
  {
    std::cout<<"[Color eval] base_color "<<base_color<<"\n";
  }

  Vec<float32,3> wh = wi + wo;
  wh.normalize();

  if(debug)
  {
    std::cout<<"[Color eval] wi "<<wi<<"\n";
    std::cout<<"[Color eval] wo "<<wo<<"\n";
  }

  float32 n_dot_l = tcos_theta(wi);
  float32 n_dot_v = tcos_theta(wo);
  float32 n_dot_h = tcos_theta(wh);
  float32 l_dot_h = dot(wi, wh);

  float32 ax,ay;
  calc_anisotropic(mat.m_roughness, mat.m_anisotropic, ax, ay);

  if(mat.m_spec_trans > 0.f)
  {

     float32 thin_roughness = mat.m_roughness * clamp(0.65f * mat.m_ior - 0.35f, 0.f, 1.f);
     if(debug)
     {
        std::cout<<"[Color eval] thin rough "<<thin_roughness<<"\n";
     }

    ax *= thin_roughness;
    ay *= thin_roughness;
    float32 ni = wo[2] > 0.0f ? 1.0f : mat.m_ior;
    float32 nt = wo[2] > 0.0f ? mat.m_ior : 1.0f;
    float32 eta = ni / nt;

    Vec<float32,3> twh = (wo + eta * wi);
    twh.normalize();
    if(n_dot_l < 0.f)
    {
      // thin transmission
      bsdf[0] = sqrt(base_color[0]);
      bsdf[1] = sqrt(base_color[1]);
      bsdf[2] = sqrt(base_color[2]);
      if(debug)
      {
        std::cout<<"[Color eval] bsdf sqrt "<<bsdf<<"\n";
        std::cout<<"[Color eval] bsdf ax ay "<<ax<<" "<<ay<<"\n";
        std::cout<<"[Color eval] bsdf twh "<<twh<<"\n";
      }
      twh[2] = -twh[2];
    }
    else
    {
      bsdf[0] = base_color[0];
      bsdf[1] = base_color[1];
      bsdf[2] = base_color[2];
      if(debug)
      {
        std::cout<<"[Color eval] bsdf using base_color "<<bsdf<<"\n";
      }
    }

    float32 d = gtr2_aniso(twh, ax, ay, debug);
    float32 gl = separable_ggx_aniso(wi, ax, ay);
    float32 gv = separable_ggx_aniso(wo, ax, ay);
    float32 f = dielectric(dot(wo,wh), 1.0f, mat.m_ior);

    //float32 c = (abs(l_dot_h) * abs(dot(wh,wo))) / (abs(n_dot_l) * abs(n_dot_v));
    float32 c = (abs(dot(wi,twh)) * abs(dot(twh,wo))) / (abs(n_dot_l) * abs(n_dot_v));
    float32 n2 = mat.m_ior * mat.m_ior;
    float32 den = l_dot_h + mat.m_ior * dot(twh,wo);
    float32 t = n2 / ( den * den);

    bsdf = bsdf * c * t * (1.f - f) * gl * gv * d;
    if(debug)
    {
      std::cout<<"[Color eval] bsdf base_color "<<bsdf<<"\n";
      std::cout<<"[Color eval] bsdf masking gl "<<gl<<"\n";
      std::cout<<"[Color eval] bsdf masking gv "<<gv<<"\n";
      std::cout<<"[Color eval] bsdf t "<<t<<"\n";
      std::cout<<"[Color eval] bsdf d "<<d<<"\n";
      std::cout<<"[Color eval] bsdf c "<<c<<"\n";
      std::cout<<"[Color eval] bsdf f "<<f<<"\n";
      std::cout<<"[Color eval] bsdf abs_l_dot_h  "<<abs(l_dot_h)<<"\n";
      std::cout<<"[Color eval] bsdf abs_v_dot_h  "<<abs(dot(wh,wo))<<"\n";
      std::cout<<"[Color eval] bsdf abs_n_dot_l  "<<abs(n_dot_l)<<"\n";
      std::cout<<"[Color eval] bsdf abs_n_dot_v  "<<abs(n_dot_v)<<"\n";
      std::cout<<"[Color eval] bsdf h_dot_n "<<tcos_theta(wh)<<"\n";
    }

  }

  if(debug)
  {
    std::cout<<"[Color eval] n_dot_l "<<n_dot_l<<"\n";
    std::cout<<"[Color eval] n_dot_v "<<n_dot_v<<"\n";
  }

  if((mat.m_spec_trans < 1.f) && (n_dot_l > 0.f) && (n_dot_v > 0.f))
  {
    float32 clum = 0.3f * base_color[0] +
                   0.6f * base_color[1] +
                   0.1f * base_color[2];

    Vec<float32,3> ctint = {{1.f, 1.f, 1.f}};
    if(clum > 0.0)
    {
      ctint = base_color / clum;
    }
    constexpr Vec<float32,3> cone = {{1.f, 1.f, 1.f}};

    Vec<float32,3> csheen = {{1.f, 1.f, 1.f}};
    csheen = mix(cone, ctint, mat.m_sheen_tint);
    Vec<float32,3> cspec = mix(mat.m_specular * 0.08f * mix(cone, ctint, mat.m_spec_tint),
                               base_color,
                               mat.m_metallic);


    // diffuse fresnel
    float32 fl = schlick_fresnel(n_dot_l);
    float32 fv = schlick_fresnel(n_dot_v);
    float32 fd90 = 0.5f + 2.0f * l_dot_h * l_dot_h * mat.m_roughness;
    float32 fd = mix(1.f, fd90, fl) * mix(1.f, fd90, fv);

    // subsurface
    float32 fss90 = l_dot_h * l_dot_h * mat.m_roughness;
    float32 fss = mix(1.f, fss90, fl) * mix(1.f, fss90, fv);
    float32 ss = 1.25f * (fss * (1.f/(n_dot_l + n_dot_v) - 0.5f) + 0.5f);

    // specular
    float32 ax,ay;
    calc_anisotropic(mat.m_roughness, mat.m_anisotropic, ax, ay);

    float32 ds = gtr2_aniso(wh, ax, ay);
    float32 fh = schlick_fresnel(l_dot_h);
    Vec<float32,3> fs = mix(cspec, cone, fh);
    float32 gl = separable_ggx_aniso(wi, ax, ay);
    float32 gv = separable_ggx_aniso(wo, ax, ay);
    float32 gs = gl * gv;

    if(debug)
    {
      std::cout<<"[Color eval] base_color "<<base_color<<"\n";
      std::cout<<"[Color eval] fs "<<fs<<"\n";
      std::cout<<"[Color eval] gs "<<gs<<"\n";
      std::cout<<"[Color eval] ds "<<ds<<"\n";
    }


    // sheen
    Vec<float32,3> fsheen =  fh * mat.m_sheen * csheen;

    // clear coat
    float32 dr = gtr1(n_dot_h, mix(0.1f, 0.001f, mat.m_clearcoat_gloss));
    float32 fr = mix(0.04f, 1.f, fh);
    float32 gr = smithg_ggx(n_dot_l, 0.25f) * smithg_ggx(n_dot_v,0.25f);

    float32 inv_pi = 1.f / pi();
    Vec<float32,3> diff = (inv_pi * mix(fd, ss, mat.m_subsurface) *  base_color +fsheen) *
                          (1.f - mat.m_metallic);

    Vec<float32,3> spec =  gs * ds * fs;
    float32 cc_fact = 0.25f * mat.m_clearcoat * gr * fr * dr;
    Vec<float32,3> clearcoat = {{cc_fact, cc_fact, cc_fact}};
    brdf = diff + spec + clearcoat;

    if(debug)
    {
      std::cout<<"[Color eval] cspec "<<cspec<<"\n";
      std::cout<<"[Color eval] spec "<<spec<<"\n";
      std::cout<<"[Color eval] diff "<<diff<<"\n";
      std::cout<<"[Color eval] clearcoat "<<clearcoat<<"\n";
    }
  }

  color = mix(brdf,bsdf, mat.m_spec_trans);
  if(debug)
  {
    std::cout<<"[Color eval] brdf "<<brdf<<"\n";
    std::cout<<"[Color eval] bsdf "<<bsdf<<"\n";
    std::cout<<"[Color eval] color "<<color<<"\n";
  }

  return color;
}


} // namespace dray
#endif
