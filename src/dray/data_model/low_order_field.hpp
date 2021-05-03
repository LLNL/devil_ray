// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_LOW_ORDER_FIELD_HPP
#define DRAY_LOW_ORDER_FIELD_HPP

#include <dray/data_model/field.hpp>
#include <dray/array.hpp>
#include <dray/lazy_prop.hpp>

namespace dray
{

class LowOrderField : public Field
{
public:
  enum class Assoc { Vertex, Element };
protected:
  Assoc m_assoc;
  Vec<int32, 3> m_cell_dims;
  Array<Float> m_values;

  struct CalcRange {
    Range operator()(const LowOrderField *) const;
  };

  LazyProp<Range, CalcRange, const LowOrderField*> m_range =
      {CalcRange(), this};

public:
  LowOrderField() = delete;
  LowOrderField(LowOrderField &other);
  LowOrderField(Array<Float> values, Assoc assoc, const Vec<int32, 3> &cell_dims);
  virtual ~LowOrderField();

  virtual std::vector<Range> range() const override;
  virtual int32 order() const override;
  virtual int32 components() const override;
  virtual std::string type_name() const override;
  virtual void to_node(conduit::Node &n_field) override;
  virtual void to_blueprint(conduit::Node &n_dataset) override;
  virtual void eval(const Array<Location> locs, Array<Float> &values) override;

  Array<Float> values();
  LowOrderField::Assoc assoc() const;
  const Vec<int32, 3> & cell_dims() const;
};

} // namespace dray

#endif // DRAY_LOW_ORDER_FIELD_HPP
