// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_LOW_ORDER_FIELD_HPP
#define DRAY_LOW_ORDER_FIELD_HPP

#include <dray/data_model/field.hpp>
#include <dray/array.hpp>

namespace dray
{

class LowOrderField : public Field
{
public:
  enum class Assoc { Vertex, Element };
protected:
  Assoc m_assoc;
  Array<Float> m_values;

  struct CalcRange {
    Range operator()(const LowOrderField *) const;
  };

  Range m_range;
  mutable bool m_range_calculated;

public:
  LowOrderField() = delete;
  LowOrderField(LowOrderField &other);
  LowOrderField(Array<Float> values, Assoc assoc);
  virtual ~LowOrderField();

  virtual std::vector<Range> range() const override;
  virtual int32 order() const override;
  virtual int32 components() const override;
  virtual std::string type_name() const override;
  virtual void to_node(conduit::Node &n_field) override;
  virtual void eval(const Array<Location> locs, Array<Float> &values) override;
  void to_blueprint(conduit::Node &n_dataset);

  Array<Float> values();
  LowOrderField::Assoc assoc() const;
};

} // namespace dray

#endif // DRAY_LOW_ORDER_FIELD_HPP
