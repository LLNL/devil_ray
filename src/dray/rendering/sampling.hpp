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

// the pdf which generated the ray direction goes first.
DRAY_EXEC
float32 power_heuristic(float32 a, float32 b)
{
  float t = a * a;
  return t / (b * b + t);
}

DRAY_EXEC
Vec3f reflect(const Vec3f &wo, const Vec3f &n)
{
  return  2.f * dot(wo, n) * n - wo;
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
Vec<float32, 3>
cosine_weighted_hemisphere (const Vec<float32,2> &xy)
{
  const float32 phi = 2.f * pi() * xy[0];
  const float32 cosTheta = sqrt(xy[1]);
  const float32 sinTheta = sqrt(1.0f - xy[1]);
  Vec<float32, 3> wo;
  wo[0] = cosTheta * cos(phi);
  wo[1] = cosTheta * sin(phi);;
  wo[2] = sinTheta;

  return wo;
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
    printf("Inside sphere\n");
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
  return point;
}


inline
void compute_distribution1d(const float32 *function, // size
                            float32 *cdf,      // size + 1
                            float32 &integral,
                            const int32 size)
{
  cdf[0] = 0;
  // step function
  for(int32 i = 1; i < size + 1; ++i)
  {
    cdf[i] = cdf[i-1] + function[i-1] / float32(size);
  }

  integral = cdf[size];
  if(integral == 0.f)
  {
    // use equal probabilites
    for(int32 i = 1; i < size + 1; ++i)
    {
      cdf[i] = float32(i) / float32(size);
    }
  }
  else
  {
    for(int32 i = 1; i < size + 1; ++i)
    {
      cdf[i] = cdf[i] / integral;
    }
  }

}


struct Distribution1D
{
  Array<float32> m_function;
  Array<float32> m_cdf;
  float32 m_integral;

  Distribution1D(){};


  Distribution1D(Array<float32> &f)
  {
    compute(f);
  }
  void compute(Array<float32> &f)
  {
    m_function = f;
    const int32 size = f.size();
    const float32 *f_ptr = f.get_host_ptr_const();
    m_cdf.resize(size+1);
    float32 *cdf_ptr = m_cdf.get_host_ptr();

    compute_distribution1d(f_ptr, cdf_ptr, m_integral, size);
  }

};

struct DeviceDistribution1D
{
  const float32 *m_cdf;
  const float32 *m_function;
  const float32 m_integral;
  const int32 m_size;

  DeviceDistribution1D(Distribution1D &dist)
    : m_cdf(dist.m_cdf.get_device_ptr_const()),
      m_function(dist.m_function.get_device_ptr_const()),
      m_integral(dist.m_integral),
      m_size(dist.m_function.size())
  {
  }

  DRAY_EXEC
  int32 discrete_sample(const float32 u, float32 &pdf, bool debug = false) const
  {
    // binary search for the interval
    const int32 cdf_size = m_size + 1;
    int32 first = 0;
    int32 len = cdf_size;
    while(len > 0)
    {
      int32 half = len >> 1;
      int32 middle = first + half;
      if(m_cdf[middle] <= u)
      {
        first = middle + 1;
        len -= half + 1;
      }
      else
      {
        len = half;
      }
    }

    int32 index = clamp(first - 1, 0, cdf_size - 2);
    pdf = 0;
    if(m_integral > 0)
    {
      pdf = m_function[index] / (m_integral * m_size);
    }
    if(debug)
    {
      printf("[Discrete sample] index %d\n",index);
      printf("[Discrete sample] integral %f\n",m_integral);
      printf("[Discrete sample] size %d\n",m_size);
      printf("[Discrete sample] f[index] %f\n",m_function[index]);
    }
    return index;
  }
}; // device distribution 1d

struct Distribution2D
{
  Array<float32> m_func;
  int32 m_width;
  int32 m_height;

  // array of (width + 1) * height where each row is a 1d distribution
  Array<float32> m_cdfs;
  // array of width (the integrals of each row)
  Array<float32> m_integrals;

  // Data for a 1d distribution build on the integrals of rows.
  Array<float32> m_y_cdf;
  float32 m_y_integral;

  Distribution2D() = default;

  Distribution2D(Array<float32> &func, const int32 width, const int32 height)
    : m_func(func),
      m_width(width),
      m_height(height)
  {
    const int32 cdf_size = (m_width + 1) * m_height;
    m_cdfs.resize(cdf_size);
    m_integrals.resize(m_height);

    float32 *func_ptr = m_func.get_host_ptr();
    float32 *cdfs_ptr = m_cdfs.get_host_ptr();
    float32 *integrals_ptr = m_integrals.get_host_ptr();
    for(int32 i = 0; i < height; ++i)
    {
      compute_distribution1d(func_ptr + i * m_width,
                             cdfs_ptr + i * (m_width + 1),
                             integrals_ptr[i],
                             m_width);
    }

    m_y_cdf.resize(m_height + 1);
    float32 *y_cdf_ptr = m_y_cdf.get_host_ptr();
    compute_distribution1d(integrals_ptr,
                           y_cdf_ptr,
                           m_y_integral,
                           height);

  }
};

struct DeviceDistribution2D
{
  const float32 *m_func;
  const int32 m_width;
  const int32 m_height;

  const float32 *m_cdfs;
  const float32 *m_integrals;

  const float32 *m_y_cdf;
  const float32 m_y_integral;

  DeviceDistribution2D(const Distribution2D &dist)
    : m_func(dist.m_func.get_device_ptr_const()),
      m_width(dist.m_width),
      m_height(dist.m_height),
      m_cdfs(dist.m_cdfs.get_device_ptr_const()),
      m_integrals(dist.m_integrals.get_device_ptr_const()),
      m_y_cdf(dist.m_y_cdf.get_device_ptr_const()),
      m_y_integral(dist.m_y_integral)

  {
  }

  // this is an internal method, but i don't think
  // I can make it private due to lambda capture issues.
  DRAY_EXEC
  float32 sample_1d(float32 u,
                    float32 &pdf,
                    int32 &index, // index of the sample
                    const int32 size,
                    const float32 *cdf,
                    const float32 *func,
                    const float32 integral) const
  {
    // binary search for the interval
    const int32 cdf_size = size + 1;
    int32 first = 0;
    int32 len = cdf_size;
    while(len > 0)
    {
      int32 half = len >> 1;
      int32 middle = first + half;

      if(cdf[middle] <= u)
      {
        first = middle + 1;
        len -= half + 1;
      }
      else
      {
        len = half;
      }
    }

    index = clamp(first - 1, 0, cdf_size - 2);
    pdf = 0;

    const float32 cval = cdf[index];
    float32 du = u - cval;
    const float32 delta = cdf[index+1] - cval;
    if(delta > 0)
    {
      du = du / delta;
    }

    pdf = integral > 0 ? func[index] / integral : 0.f;
    return (float32(index) + du) / float32(size);
  }

  // returns values in [0,1) for each dim along with the pdf
  DRAY_EXEC
  Vec<float32,2> sample(const Vec<float32,2> &rand, float32 &pdf) const
  {
    Vec<float32,2> res;
    // sample the y
    float32 pdf_y, pdf_x;
    int32 y_index;
    res[1] = sample_1d(rand[1], pdf_y, y_index, m_height, m_y_cdf, m_integrals, m_y_integral);

    int32 x_index;
    res[0] = sample_1d(rand[0],
                       pdf_x,
                       x_index,
                       m_width,
                       m_cdfs + y_index * (m_width + 1),
                       m_func + y_index * m_width,
                       m_integrals[y_index]);
    (void) x_index;
    pdf = pdf_x * pdf_y;
    return res;

  }

};

} // namespace dray
#endif
