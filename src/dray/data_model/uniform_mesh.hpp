// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_UNIFORM_MESH_HPP
#define DRAY_UNIFORM_MESH_HPP

#include <dray/data_model/mesh.hpp>

namespace dray
{

class UniformMesh : public Mesh
{

protected:
  Vec<Float, 3> m_spacing;
  Vec<Float, 3> m_origin;
  Vec<int32, 3> m_dims;
public:
  UniformMesh() = delete;
  UniformMesh(const Vec<Float,3> &spacing,
                  const Vec<Float,3> &m_origin,
                  const Vec<int32,3> &dims);

  virtual ~UniformMesh();
  virtual int32 cells() const override;

  virtual int32 order() const override;

  virtual int32 dims() const override;

  Vec<int32,3> cell_dims() const;
  Vec<Float,3> spacing() const;
  Vec<Float,3> origin() const;

  virtual std::string type_name() const override;

  // bounds() should be const, but DerivedTopology/Mesh needs mutable.
  virtual AABB<3> bounds() override;

  // locate() should be const, but DerivedTopology/Mesh needs mutable.
  virtual Array<Location> locate (Array<Vec<Float, 3>> &wpoints) override;

  virtual void to_node(conduit::Node &n_topo) override;

  //virtual void to_blueprint(conduit::Node &n_dataset) override;
  void to_blueprint(conduit::Node &n_dataset);

  friend struct UniformDeviceMesh;
};

} // namespace dray

#endif // DRAY_REF_POINT_HPP
