// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_UNIFORM_TOPOLOGY_HPP
#define DRAY_UNIFORM_TOPOLOGY_HPP

#include <dray/data_model/mesh.hpp>

namespace dray
{

class UniformTopology : public Mesh, public Structured
{

protected:
  Vec<Float, 3> m_spacing;
  Vec<Float, 3> m_origin;
  Vec<int32, 3> m_dims;
public:
  UniformTopology() = delete;
  UniformTopology(const Vec<Float,3> &spacing,
                  const Vec<Float,3> &m_origin,
                  const Vec<int32,3> &dims);

  virtual ~UniformTopology();
  virtual int32 cells() const override;

  virtual int32 order() const override;

  virtual int32 dims() const override;

  virtual Vec<int32,3> cell_dims() const override;
  Vec<Float,3> spacing() const;
  Vec<Float,3> origin() const;

  virtual std::string type_name() const override;

  // bounds() should be const, but DerivedTopology/Mesh needs mutable.
  virtual AABB<3> bounds() override;

  // locate() should be const, but DerivedTopology/Mesh needs mutable.
  virtual Array<Location> locate (Array<Vec<Float, 3>> &wpoints) override;

  Location locate(const Vec<Float, 3> &wpt) const;

  virtual void to_node(conduit::Node &n_topo) override;

  virtual void to_blueprint(conduit::Node &n_dataset) override;


  struct Evaluator
  {
    Vec<Float, 3> m_spacing;
    Vec<Float, 3> m_origin;
    Vec<int32, 3> m_dims;

    DRAY_EXEC Vec<Float, 3> operator()(const Location &loc) const;
  };

  struct JacobianEvaluator
  {
    Vec<Vec<Float, 3>, 3> m_val;

    DRAY_EXEC Vec<Vec<Float, 3>, 3> operator()(const Location &) const
    {
      return m_val;
    }
  };

  Evaluator evaluator() const;
  JacobianEvaluator jacobian_evaluator() const;

private:
  AABB<3> bounds_const() const;
};


DRAY_EXEC Vec<Float, 3> UniformTopology::Evaluator::operator()(
    const Location &loc) const
{
  int32 cell = loc.m_cell_id;
  Vec<Float, 3> xyz;
  xyz[0] = cell % m_dims[0];
  cell /= m_dims[0];
  xyz[1] = cell % m_dims[1];
  cell /= m_dims[1];
  xyz[2] = cell;// % m_dims[2];

  xyz += loc.m_ref_pt;

  xyz[0] *= m_spacing[0];
  xyz[1] *= m_spacing[1];
  xyz[2] *= m_spacing[2];

  xyz += m_origin;

  return xyz;
}



namespace detail
{
  DRAY_EXEC Location uniform_locate_float(
      const Vec<Float, 3> &xyz,  // xyz relative to origin
      const Vec<int32, 3> &dims,
      const Vec<Float, 3> &spacing);

  DRAY_EXEC Location uniform_locate_int(
      Vec<int32, 3> ijk,
      const Vec<int32, 3> &dims);
}

} // namespace dray


namespace dray
{
  namespace detail
  {
    //
    // uniform_locate_float()
    //
    DRAY_EXEC Location uniform_locate_float(
        const Vec<Float, 3> &xyz,
        const Vec<int32, 3> &dims,
        const Vec<Float, 3> &spacing)
    {
      Vec<Float, 3> scaled;
      scaled[0] = xyz[0] / spacing[0];
      scaled[1] = xyz[1] / spacing[1];
      scaled[2] = xyz[2] / spacing[2];

      Vec<int32, 3> scaled_int;
      scaled_int[0] = (int32) scaled[0];
      scaled_int[1] = (int32) scaled[1];
      scaled_int[2] = (int32) scaled[2];

      // Integer location, missing fractional ref pt.
      Location loc = uniform_locate_int(scaled_int, dims);

      // Add back the fractional part.
      loc.m_ref_pt[0] += scaled[0] - scaled_int[0];
      loc.m_ref_pt[1] += scaled[1] - scaled_int[1];
      loc.m_ref_pt[2] += scaled[2] - scaled_int[2];

      return loc;
    }

    //
    // uniform_locate_int()
    //
    DRAY_EXEC Location uniform_locate_int(
        Vec<int32, 3> ijk,
        const Vec<int32, 3> &dims)
    {
      Location loc = {0, {0, 0, 0}};

      // Clamp
      if (ijk[0] == Float(dims[0]))
      {
        ijk[0] = dims[0] - 1;
        loc.m_ref_pt[0] = 1.0f;
      }
      if (ijk[1] == Float(dims[1]))
      {
        ijk[1] = dims[1] - 1;
        loc.m_ref_pt[1] = 1.0f;
      }
      if (ijk[2] == Float(dims[2]))
      {
        ijk[2] = dims[2] - 1;
        loc.m_ref_pt[2] = 1.0f;
      }

      loc.m_cell_id = (ijk[2] * dims[1] + ijk[1]) * dims[0] + ijk[0];

      return loc;
    }

  }// namespace detail
}// namespace dray

#endif // DRAY_REF_POINT_HPP
