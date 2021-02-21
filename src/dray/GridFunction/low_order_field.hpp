// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_LOW_ORDER_FIELD_HPP
#define DRAY_LOW_ORDER_FIELD_HPP

#include <dray/GridFunction/field_base.hpp>
#include <dray/array.hpp>
#include <dray/lazy_prop.hpp>

namespace dray
{

class LowOrderField : public FieldBase
{
public:
  enum class Assoc { Vertex, Element };
  struct CalcRange {
    Range operator()(const LowOrderField *) const;
  };
protected:
  Assoc m_assoc;
  Array<Float> m_values;

  LazyProp<Range, CalcRange, const LowOrderField*> m_range =
      {CalcRange(), this};

public:
  LowOrderField() = delete;
  LowOrderField(Array<Float> values, Assoc assoc);
  virtual ~LowOrderField();

  virtual std::vector<Range> range() const override;
  virtual int32 order() const override;
  virtual std::string type_name() const override;
  virtual void to_node(conduit::Node &n_field) override;

  Array<Float> values();
  LowOrderField::Assoc assoc() const;

};

} // namespace dray

#endif // DRAY_LOW_ORDER_FIELD_HPP
