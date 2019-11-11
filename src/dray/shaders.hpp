#ifndef DRAY_SHADERS_HPP
#define DRAY_SHADERS_HPP

#include <dray/array.hpp>
#include <dray/color_table.hpp>
#include <dray/color_map.hpp>
#include <dray/fragment.hpp>
#include <dray/framebuffer.hpp>
#include <dray/ray.hpp>
#include <dray/ray_hit.hpp>
#include <dray/vec.hpp>
#include <dray/math.hpp>

namespace dray
{

struct PointLightSource
{
  Vec<float32,3> m_pos;
  Vec<float32,3> m_amb;
  Vec<float32,3> m_diff;
  Vec<float32,3> m_spec;
  float32 m_spec_pow;
};

class ShadeMeshLines
{
  protected:
    Vec4f u_edge_color;
    Vec4f u_face_color;
    float32 u_edge_radius_rcp;
    int32 u_grid_res;

  public:
    void set_uniforms(Vec4f edge_color, Vec4f face_color, float32 edge_radius, int32 grid_res = 1)
    {
      u_edge_color = edge_color;
      u_face_color = face_color;
      u_edge_radius_rcp = (edge_radius > 0.0 ? 1.0/edge_radius : 0.05) / grid_res;
      u_grid_res = grid_res;
    }

    DRAY_EXEC Vec4f operator()(const Vec<Float,2> &rcoords) const
    {
      // Get distance to nearest edge.
      float32 edge_dist = 0.0;
      {
        Vec<Float,2> prcoords = rcoords;
        prcoords[0] = u_grid_res * prcoords[0];  prcoords[0] -= floor(prcoords[0]);
        prcoords[1] = u_grid_res * prcoords[1];  prcoords[1] -= floor(prcoords[1]);

        float32 d0 = (prcoords[0] < 0.0 ? 0.0 : prcoords[0] > 1.0 ? 0.0 : 0.5 - fabs(prcoords[0] - 0.5));
        float32 d1 = (prcoords[1] < 0.0 ? 0.0 : prcoords[1] > 1.0 ? 0.0 : 0.5 - fabs(prcoords[1] - 0.5));

        float32 min2 = (d0 < d1 ? d0 : d1);
        edge_dist = min2;
      }
      edge_dist *= u_edge_radius_rcp;  // Normalized distance from nearest edge.
      // edge_dist is nonnegative.

      const float32 x = min(edge_dist, 1.0f);

      // Cubic smooth interpolation.
      float32 w = (2.0 * x - 3.0) * x * x + 1.0;
      Vec4f frag_color = u_edge_color * w + u_face_color * (1.0-w);
      return frag_color;
    }

    DRAY_EXEC Vec4f operator()(const Vec<Float,3> &rcoords) const
    {
      // Since it is assumed one of the coordinates is 0.0 or 1.0 (we have a face point),
      // we want to measure the second-nearest-to-edge distance.

      float32 edge_dist = 0.0;
      {
        Vec<Float,3> prcoords = rcoords;
        prcoords[0] = u_grid_res * prcoords[0];  prcoords[0] -= floor(prcoords[0]);
        prcoords[1] = u_grid_res * prcoords[1];  prcoords[1] -= floor(prcoords[1]);
        prcoords[2] = u_grid_res * prcoords[2];  prcoords[2] -= floor(prcoords[2]);

        float32 d0 = (prcoords[0] < 0.0 ? 0.0 : prcoords[0] > 1.0 ? 0.0 : 0.5 - fabs(prcoords[0] - 0.5));
        float32 d1 = (prcoords[1] < 0.0 ? 0.0 : prcoords[1] > 1.0 ? 0.0 : 0.5 - fabs(prcoords[1] - 0.5));
        float32 d2 = (prcoords[2] < 0.0 ? 0.0 : prcoords[2] > 1.0 ? 0.0 : 0.5 - fabs(prcoords[2] - 0.5));

        float32 min2 = (d0 < d1 ? d0 : d1);
        float32 max2 = (d0 < d1 ? d1 : d0);
        // Now three cases: d2 < min2 <= max2;   min2 <= d2 <= max2;   min2 <= max2 < d2;
        edge_dist = (d2 < min2 ? min2 : max2 < d2 ? max2 : d2);
      }
      edge_dist *= u_edge_radius_rcp;  // Normalized distance from nearest edge.
      // edge_dist is nonnegative.

      const float32 x = min(edge_dist, 1.0f);

      // Cubic smooth interpolation.
      float32 w = (2.0 * x - 3.0) * x * x + 1.0;
      Vec4f frag_color = u_edge_color * w + u_face_color * (1.0-w);
      return frag_color;
    }
};

class Shader
{
public:
  static void composite_bg(dray::Array<dray::Vec<float, 4> > &color_buffer,
                           const dray::Vec<float, 4> &bg_color);


static void blend(Framebuffer &fb,
                  ColorMap &color_map,
                  const Array<Ray> &rays,
                  const Array<RayHit> &hits,
                  const Array<Fragment> &fragments);

#warning "shaders are making less sense with rayhits, locations and others."
#warning "Unifiy or move into filters/framebuffer"

static void blend_surf(Framebuffer &fb,
                       ColorMap &color_map,
                       const Array<Ray> &rays,
                       const Array<RayHit> &hits,
                       const Array<Fragment> &fragments);

static void set_color_table(const ColorTable &color_table);

static int32 m_color_samples;

static void set_light_properties(const PointLightSource &light) { m_light = light; }
static void set_light_position(const Vec<float32,3> &pos) { m_light.m_pos = pos; }

private:
  //static Array<Vec4f> m_color_map;  // As a static member, was causing problems upon destruction.
  static ColorTable m_color_table;    // This is not an array, so should be fine.

  // Light properties.
  static PointLightSource m_light;
};

} // namespace dray

#endif
