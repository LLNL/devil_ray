// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_TOPOLOGY_BASE_HPP
#define DRAY_TOPOLOGY_BASE_HPP

#include <dray/GridFunction/field.hpp>
#include <dray/GridFunction/mesh.hpp>

namespace dray
{

class TopologyBase
{
protected:
  std::string m_name;
  std::string m_shape_name;
public:
  virtual ~TopologyBase(){};

  std::string name() const;
  void name(const std::string &name);

  virtual std::string type_name() const = 0;
  virtual int32 cells() const = 0;
  virtual int32 order() const = 0;
  virtual int32 dims() const = 0;
  virtual AABB<3> bounds() const = 0;
  virtual Array<Location> locate (Array<Vec<Float, 3>> &wpoints) const = 0;
  virtual Array<Vec<Float, 3>> eval_location (Array<Location> &rpoints) const = 0;
};

} // namespace dray

#endif // DRAY_TOPOLGY_BASE_HPP
