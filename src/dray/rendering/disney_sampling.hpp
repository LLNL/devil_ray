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
float32 scale_roughness(const float32 roughness, const float32 ior)
{
    return roughness * clamp(0.65f * ior - 0.35f, 0.f, 1.f);
}

DRAY_EXEC
bool same_hemi(const Vec<float32,3> &w1, const Vec<float32,3> &w2)
{
  return w1[2] * w2[2] > 0.f;
}

DRAY_EXEC
Vec<float32,3> refract(const Vec<float32,3> &wi,
                       const Vec<float32,3> &n,
                       float32 eta,
                       bool &valid)
{
     Vec<float32,3> wt;
    // Compute $\cos \theta_\roman{t}$ using Snell's law
    float32 cos_theta_i = dot(n, wi);
    float32 sin2_theta_i = max(0.f, 1.f - cos_theta_i * cos_theta_i);
    float32 sin2_theta_t = eta * eta * sin2_theta_i;

    // Handle total internal reflection for transmission
    if (sin2_theta_t >= 1.f) valid = false;
    float32 cos_theta_t = sqrt(1 - sin2_theta_t);
    wt = eta * -wi + (eta * cos_theta_i - cos_theta_t) * n;
    return wt;
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
float32 lambda(const Vec<float32,3> &w,
               const float32 ax,
               const float32 ay)
{

  if(tcos_theta(w) == 0.f)
  {
    return 0.f;
  }

  float32 abs_tan_theta = abs(ttan_theta(w));
  float32 alpha = sqrt(tcos2_phi(w) * ax * ax + tsin2_phi(w) * ay * ay);
  float32 atan_theta = alpha * abs_tan_theta * alpha * abs_tan_theta;
  return 0.5f * (-1.f + sqrt(1.f + atan_theta));
}

DRAY_EXEC
float32 ggx_g(const Vec<float32,3> &wo,
              const Vec<float32,3> &wi,
              const float32 ax,
              const float32 ay)
{
  return 1.f / (1.f + lambda(wi, ax, ay) + lambda(wo, ax, ay));
}

float32 ggx_g1(const Vec<float32,3> &w,
               const float32 ax,
               const float32 ay)
{
  return 1.f / (1.f + lambda(w, ax, ay));
}

float32 ggx_d(const Vec<float32,3> &wh, const float32 ax, const float32 ay)
{
  if(tcos_theta(wh) == 0.f)
  {
    return 0.f;
  }

  float32 tan2_theta = ttan2_theta(wh);

  float32 cos4_theta = tcos2_theta(wh) * tcos2_theta(wh);
  float32 e = (tcos2_phi(wh) / (ax * ax) + tsin2_phi(wh) / (ay * ay) ) * tan2_theta;
  return 1.f / (pi() * ax * ay * cos4_theta * (1.f + e) * (1.f + e));
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
  float32 h_dot_x = wh[0];
  float32 h_dot_y = wh[1];
  float32 n_dot_h = tcos_theta(wh);

  float32 a = h_dot_x / ax;
  float32 b = h_dot_y / ay;
  float32 c = a * a + b * b + n_dot_h * n_dot_h;
  if(debug)
  {
    std::cout<<"[gtr2] a b c "<<a<<" "<<b<<" "<<c<<"\n";
  }

  return 1.0f / (pi() * ax * ay * c * c);
}

float32 dielectric(float32 cos_theta_i, float32 ni, float32 nt)
{
    // Copied from PBRT. This function calculates the
    // full Fresnel term for a dielectric material.

    cos_theta_i = clamp(cos_theta_i, -1.0f, 1.0f);

    // Swap index of refraction if this is coming from inside the surface
    if(cos_theta_i < 0.0f)
    {
      float32 temp = ni;
      ni = nt;
      nt = temp;
      cos_theta_i = -cos_theta_i;
    }

    float32 sin_theta_i = sqrtf(max(0.0f, 1.0f - cos_theta_i * cos_theta_i));
    float32 sin_theta_t = ni / nt * sin_theta_i;

    // Check for total internal reflection
    if(sin_theta_t >= 1) {
        return 1;
    }

    float32 cos_theta_t = sqrtf(max(0.0f, 1.0f - sin_theta_t * sin_theta_t));

    float32 r_parallel = ((nt * cos_theta_i) - (ni * cos_theta_t)) / ((nt * cos_theta_i) + (ni * cos_theta_t));
    float32 r_perpendicuar = ((ni * cos_theta_i) - (nt * cos_theta_t)) / ((ni * cos_theta_i) + (nt * cos_theta_t));
    return (r_parallel * r_parallel + r_perpendicuar * r_perpendicuar) / 2;
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
  // http://www.jcgt.org/published/0007/04/01/paper.pdf

  // stretched view vector
  Vec<float32,3> s_view;
  s_view[0] = wo[0] * ax;
  s_view[1] = wo[1] * ay;
  s_view[2] = wo[2];
  s_view.normalize();


  Vec<float32,3> wcX, wcY;
  create_basis(s_view,wcX,wcY);

  float32 r = sqrt(rand[0]);
  float32 phi = 2.f * pi() * rand[1];
  float32 t1 = r * cos(phi);
  float32 t2 = r * sin(phi);
  float32 s = 0.5f * (1.f + s_view[2]);
  t2 = (1.f - s) * sqrt(1.f - t1 * t1) + s * t2;
  float32 t3 = sqrt(max(0.f, 1.f - t1 * t1 - t2 * t2));

  //// dir is the half vector
  Vec<float32,3> dir;

  dir = t1 * wcX + t2 * wcY + t3 * s_view;
  dir[0] *= ax;
  dir[1] *= ay;
  dir[2] = max(0.f, dir[2]);
  dir.normalize();

  return dir;
}

DRAY_EXEC
float32 pdf_vndf_ggx(const Vec<float32,3> &wo,
                     const Vec<float32,3> &wh,
                     const float32 ax,
                     const float32 ay,
                     bool debug = false)
{

  float32 g = ggx_g1(wo, ax, ay);

  float32 d = ggx_d(wh, ax, ay);

  if(debug)
  {
    std::cout<<"[ VNDF pdf ] g "<<g<<"\n";
    std::cout<<"[ VNDF pdf ] d "<<d<<"\n";
  }

  return g * abs(dot(wo,wh)) * d / abs(tcos_theta(wo));
}

DRAY_EXEC
Vec<float32,3> sample_microfacet_transmission(const Vec<float32,3> &wo,
                                              const float32 &eta,
                                              const float32 &ax,
                                              const float32 &ay,
                                              Vec<uint,2> &rand_state,
                                              bool &valid,
                                              bool debug = false)
{
  valid = true;
  if(wo[2] == 0.f)
  {
    valid = false;
  }

  Vec<float32,2> rand;
  rand[0] = randomf(rand_state);
  rand[1] = randomf(rand_state);
  Vec<float32,3> wh = sample_vndf_ggx(wo, ax, ay, rand);
  if(debug)
  {
    std::cout<<"[Sample MT] wh "<<wh<<"\n";
  }

  if(dot(wo, wh) < 0)
  {
    valid = false;
  }

  // normally we would calculate the eta based on the
  // side of of the wo, but we are currently modeling that
  // only thin (entrance and exit in the same interaction)

  Vec<float32,3> wi = refract(wo, wh, eta, valid);
  return wi;
}

DRAY_EXEC
float32 pdf_microfacet_transmission(const Vec<float32,3> &wo,
                                    const Vec<float32,3> &wi,
                                    float32 eta,
                                    const float32 &ax,
                                    const float32 &ay,
                                    bool debug = false)
{
  float32 pdf = 1.f;

  if(same_hemi(wo,wi))
  {
    return 0.f;
  }

  if(tcos_theta(wo) > 0.f)
  {
    eta = 1.f / eta;
  }

  Vec<float32,3> wh = wo + eta * wi;
  wh.normalize();

  if(dot(wo,wh) * dot(wi,wh) > 0.f)
  {
    pdf = 0.f;
  }

  float32 a = dot(wo,wh) + eta * dot(wi,wh);

  float32 dwh_dwi = abs((eta * eta * dot(wi,wh)) / (a * a));

  float32 distribution_pdf = pdf_vndf_ggx(wo, wh, ax, ay, debug);
  if(debug)
  {
    std::cout<<"[MT PDF] a "<<a<<"\n";
    std::cout<<"[MT PDF] dist "<<distribution_pdf<<"\n";
    std::cout<<"[MT PDF] dwf_dwi  "<<dwh_dwi<<"\n";
    std::cout<<"[MT PDF] wh "<<wh<<"\n";
  }
  pdf *= distribution_pdf * dwh_dwi;
  return pdf;
}

DRAY_EXEC
Vec<float32,3> eval_microfacet_transmission(const Vec<float32,3> &wo,
                                            const Vec<float32,3> &wi,
                                            const float32 ior,
                                            const float32 &ax,
                                            const float32 &ay,
                                            bool debug = false)
{
  Vec<float32,3> color = {{1.f, 1.f, 1.f}};

  // same hemi
  if(tcos_theta(wo) > 0.f && tcos_theta(wi) > 0.f)
  {
    color = {{0.f, 0.f, 0.f}};
  }

  float32 n_dot_v = tcos_theta(wo);
  float32 n_dot_l = tcos_theta(wi);
  if(n_dot_v == 0.f || n_dot_l == 0.f)
  {
    color = {{0.f, 0.f, 0.f}};
  }
  // flip eta if we were not just modeling thin

  // always air
  float32 eta = ior / 1.f;

  if(n_dot_v > 0.f)
  {
    eta = 1.f / eta;
  }

  Vec<float32,3> wh = wo + wi * eta;
  wh.normalize();
  // make sure we are in the same hemi as the normal
  if(wh[2] < 0)
  {
    wh = -wh;
  }

  if(dot(wo,wh) * dot(wi,wh) > 0)
  {
    color = {{0.f, 0.f, 0.f}};
  }

  float32 f = dielectric(dot(wo,wh), 1.0f, ior);

  float32 a = dot(wo,wh) + eta * dot(wi, wh);

  float32 d = ggx_d(wh, ax, ay);
  float32 g = ggx_g(wo,wi, ax, ay);
  if(debug)
  {
    std::cout<<"[Eval MT] wo "<<wo<<"\n";
    std::cout<<"[Eval MT] wi "<<wi<<"\n";
    std::cout<<"[Eval MT] eta "<<eta<<"\n";
    std::cout<<"[Eval MT] g "<<g<<"\n";
    std::cout<<"[Eval MT] d "<<d<<"\n";
    std::cout<<"[Eval MT] frensel "<<f<<"\n";
    std::cout<<"[Eval MT] wh "<<wh<<"\n";
  }

  color = color * (1.f - f) * abs(d * g *
          eta *eta * abs(dot(wi,wh)) * abs(dot(wo,wh)) /
          (n_dot_v * n_dot_l * a * a));

  return color;
}


Vec<float32,3> sample_microfacet_reflection(const Vec<float32,3> &wo,
                                            const float32 &ax,
                                            const float32 &ay,
                                            Vec<uint,2> &rand_state,
                                            bool &valid,
                                            bool debug = false)
{
  valid = true;
  // TODO: we can probaly elimate the valid
  // checks since the pdf will be 0. Check to see if
  // the pdf is zero and set the color to 0;
  if(wo[2] == 0.f)
  {
    valid = false;
  }

  Vec<float32,2> rand;
  rand[0] = randomf(rand_state);
  rand[1] = randomf(rand_state);
  //rand = {{0.994541, 0.438798}};
  Vec<float32,3> wh = sample_vndf_ggx(wo, ax, ay, rand);
  if(debug)
  {
    std::cout<<"[Sample MR] wh "<<wh<<"\n";
    std::cout<<"[Sample MR] w0 "<<wo<<"\n";
    std::cout<<"[Sample MR] ax ay "<<ax<<" "<<ay<<"\n";
    std::cout<<"[Sample MR] rand "<<rand<<"\n";
  }
  if(dot(wo,wh) < 0.)
  {
    if(debug) std::cout<<"Bad wh sample\n";
    valid = false;
  }

  Vec<float32,3> wi = reflect(wo,wh);
  if(!same_hemi(wo,wi))
  {
    if(debug) std::cout<<"Bad reflect wi "<<wi<<"\n";
    valid = false;
  }
  return wi;
}

DRAY_EXEC
Vec<float32,3> eval_microfacet_reflection(const Vec<float32,3> &wo,
                                          const Vec<float32,3> &wi,
                                          const float32 ior,
                                          const float32 &ax,
                                          const float32 &ay)
{
  Vec<float32,3> color = {{1.f, 1.f, 1.f}};

  float32 abs_n_dot_v = abs(tcos_theta(wo));
  float32 abs_n_dot_l = abs(tcos_theta(wi));
  Vec<float32,3> wh = wi + wo;
  if(abs_n_dot_v == 0.f || abs_n_dot_v == 0.f)
  {
    color = {{0.f, 0.f, 0.f}};
  }
  if(wh[0] == 0.f && wh[1] == 0.f && wh[2] == 0.f)
  {
    color = {{0.f, 0.f, 0.f}};
  }
  wh.normalize();

  float32 d = ggx_d(wh, ax, ay);
  float32 g = ggx_g(wo,wi, ax, ay);

  // for fresnel make sure that wh is in the same hemi as the normal
  // I don't this should happen but better be safe
  if(tcos_theta(wh) < 0)
  {
    wh = -wh;
  }

  float32 f = dielectric(dot(wo,wh), 1.0f, ior);

  color = color * f * d * g / (4.f * abs_n_dot_v * abs_n_dot_l);

  return color;
}

DRAY_EXEC
float32 pdf_microfacet_reflection(const Vec<float32,3> &wo,
                                  const Vec<float32,3> &wi,
                                  const float32 &ax,
                                  const float32 &ay,
                                  bool debug = false)
{
  float32 pdf = 1.f;

  if(!same_hemi(wo,wi))
  {
    return  0.f;
  }


  Vec<float32,3> wh = wo + wi;
  wh.normalize();

  float32 distribution_pdf = pdf_vndf_ggx(wo, wh, ax, ay, debug);

  pdf *= distribution_pdf / (4.0 * dot(wo,wh));
  if(debug)
  {
    std::cout<<"[MR PDF] dist "<<distribution_pdf<<"\n";
    std::cout<<"[MR PDF] wh "<<wh<<"\n";
    std::cout<<"[MR PDF] pdf "<<pdf<<"\n";
  }
  return pdf;
}

DRAY_EXEC
Vec<float32,3> sample_spec_trans(const Vec<float32,3> &wo,
                                 const Material &mat,
                                 bool &specular,
                                 Vec<uint,2> &rand_state,
                                 bool &valid,
                                 bool debug = false)
{
  valid = true;
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

  float32 theta = tcos_theta(wo);
  float32 cos2theta = 1.f - mat.m_ior * mat.m_ior * (1.f - theta * theta);
  float32 v_dot_h = dot(wo,wh);
  float32 f = dielectric(v_dot_h, 1.0f, mat.m_ior);

  if(debug)
  {
    std::cout<<"[Sample] transmission\n";
  }

  if(cos2theta < 0.f || randomf(rand_state) < f)
  {
    wi = reflect(wo,wh);
    if(!same_hemi(wo,wi))
    {
      valid = false;
    }
    if(debug)
    {
      std::cout<<"[Sample] refect\n";
    }
  }
  else
  {
    wi = refract(wo, wh, mat.m_ior, valid);
    if(dot(wh,wo)< 0.f)
    {
      valid = false;
    }
    //wi[2] = -wi[o];
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

DRAY_EXEC
float32 disney_pdf(const Vec<float32,3> &wo,
                   const Vec<float32,3> &wi,
                   const Material &mat,
                   bool debug = false)
{
  Vec<float32,3> wh = wo + wi;
  wh.normalize();

  float32 ax,ay;
  calc_anisotropic(mat.m_roughness, mat.m_anisotropic, ax, ay);
  float32 scale = mat.m_roughness * clamp(0.65f * mat.m_ior - 0.35f, 0.f, 1.f);

  float32 n_dot_h = tcos_theta(wh);

  if(debug)
  {
    std::cout<<"[PDF] n_dot_l "<<tcos_theta(wi)<<"\n";
  }

  if(!same_hemi(wo,wi))
  {
    float32 trans_pdf = pdf_microfacet_transmission(wo,wi,mat.m_ior, ax * scale, ay * scale, debug);

    float32 eta = mat.m_ior;
    if(tcos_theta(wo) > 0.f)
    {
      eta = 1.f / eta;
    }

    Vec<float32,3> wht = wo + eta * wi;
    wht.normalize();
    float32 f = dielectric(dot(wo,wht), 1.0f, mat.m_ior);

    // i feel like we have to weight by the chance of sampling this
    trans_pdf *= mat.m_spec_trans * (1.f - f);

    if(debug)
    {
      std::cout<<"[PDF] trans "<<trans_pdf<<"\n";
    }
    return trans_pdf;
  }

  float32 specular_alpha = max(0.001f, mat.m_roughness);

  float32 diff_prob = 1.f - mat.m_metallic;
  float32 spec_prob = mat.m_metallic;

  // visible normal importance sampling pdf
  float32 vndf_pdf = pdf_vndf_ggx(wo, wi, ax, ay);

  // clearcloat pdf
  float32 clearcoat_alpha = mix(0.1f,0.001f, mat.m_clearcoat_gloss);
  float32 clearcoat_pdf = gtr1(n_dot_h, clearcoat_alpha) * n_dot_h / (4.f * abs(dot(wh,wi)));
  float32 mix_ratio = 1.f / (1.f + mat.m_clearcoat);
  float32 spec_r_pdf = pdf_microfacet_reflection(wo,wi,ax,ay,debug);
  float32 pdf_spec = mix(clearcoat_pdf, spec_r_pdf, mix_ratio);

  // diffuse pdf
  float32 pdf_diff = tcos_theta(wi) / pi();

  // total brdf pdf
  float32 brdf_pdf = diff_prob * pdf_diff + spec_prob * pdf_spec;

  // bsdf reflection
  float32 bsdf_pdf = pdf_microfacet_reflection(wo, wi, ax*scale, ay*scale, debug);

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
Vec<float32,3> sample_disney(const Vec<float32,3> &wo,
                             const Material &mat,
                             bool &specular,
                             Vec<uint,2> &rand_state,
                             bool debug = false)
{

  if(debug)
  {
    std::cout<<"[Sample] mat rough "<<mat.m_roughness<<"\n";
    std::cout<<"[Sample] mat spec "<<mat.m_specular<<"\n";
    std::cout<<"[Sample] mat metallic "<<mat.m_metallic<<"\n";
  }

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
    bool valid;
    wi = sample_spec_trans(wo, mat, specular, rand_state, valid, debug);
    if(debug && !valid)
    {
      std::cout<<"[Sample] trans invalid\n";
    }
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
      bool valid;
      wi = sample_microfacet_reflection(wo, ax, ay, rand_state, valid, debug);
      specular = true;
      if(debug)
      {
        std::cout<<"[Sample] specular\n";
        if(!valid) std::cout<<"[Sample] invalid\n";
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

    //Vec<float32,3> spec =  gs * ds * fs;
    // TODO: is this right? and I need to get rid of gs,fs.ds.
    Vec<float32,3> spec =  eval_microfacet_reflection(wo,wi,mat.m_ior, ax, ay);
    spec[0] *= cspec[0];
    spec[1] *= cspec[1];
    spec[2] *= cspec[2];

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

  if(mat.m_spec_trans > 0.f)
  {
    float32 scale = mat.m_roughness * clamp(0.65f * mat.m_ior - 0.35f, 0.f, 1.f);
    Vec<float32,3> trans = eval_microfacet_transmission(wo,wi,mat.m_ior, ax * scale, ay * scale);
    trans[0] *= sqrt(base_color[0]);
    trans[1] *= sqrt(base_color[1]);
    trans[2] *= sqrt(base_color[2]);

    Vec<float32,3> ref = eval_microfacet_reflection(wo,wi,mat.m_ior, ax * scale, ay * scale);
    ref[0] *= base_color[0];
    ref[1] *= base_color[1];
    ref[2] *= base_color[2];
    bsdf = trans + ref;
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
