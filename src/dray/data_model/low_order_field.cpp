// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <dray/data_model/low_order_field.hpp>
#include <dray/policies.hpp>

namespace dray
{


Range LowOrderField::CalcRange::operator()(const LowOrderField * arg) const
{
  Range result;

  const Float *values_ptr = arg->m_values.get_device_ptr_const();
  const int32 size = arg->m_values.size();

  RAJA::ReduceMin<reduce_policy, Float> xmin (infinity<Float>());
  RAJA::ReduceMax<reduce_policy, Float> xmax (neg_infinity<Float>());

  RAJA::forall<for_policy> (RAJA::RangeSegment (0, size), [=] DRAY_LAMBDA (int32 ii)
  {
    const Float value = values_ptr[ii];
    xmin.min (value);
    xmax.max (value);
  });
  result.include (xmin.get ());
  result.include (xmax.get ());

  return result;
}


LowOrderField::LowOrderField(LowOrderField &other)
  : m_assoc(other.m_assoc),
    m_values(other.m_values),
    m_cell_dims(other.m_cell_dims)
{
  this->name(other.name());
  this->mesh_name(other.mesh_name());
}

LowOrderField::LowOrderField(Array<Float> values, Assoc assoc, const Vec<int32, 3> &cell_dims)
  : m_assoc(assoc),
    m_values(values),
    m_cell_dims(cell_dims)
{

}

LowOrderField::~LowOrderField()
{

}

Array<Float>
LowOrderField::values()
{
  return m_values;
}

LowOrderField::Assoc
LowOrderField::assoc() const
{
  return m_assoc;
}

const Vec<int32, 3> & LowOrderField::cell_dims() const
{
  return m_cell_dims;
}

std::vector<Range> LowOrderField::range() const
{
  std::vector<Range> ranges;
  ranges.push_back(m_range.get());
  return ranges;
}

int32 LowOrderField::order() const
{
  if(m_assoc == Assoc::Vertex)
  {
    return 1;
  }
  else
  {
    return 0;
  }
}

std::string LowOrderField::type_name() const
{
  std::string name = "low_order_";
  if(m_assoc == Assoc::Vertex)
  {
    name += "vertex";
  }
  else
  {
    name += "element";
  }
  return name;
}

void LowOrderField::to_node(conduit::Node &n_field)
{
  n_field.reset();
  n_field["type_name"] = type_name();
  /// n_field["order"] = get_poly_order();

  throw std::logic_error(("Not implemented to_node()! " __FILE__));

  /// conduit::Node &n_gf = n_field["grid_function"];
  /// GridFunction<ElemT::get_ncomp ()> gf = get_dof_data();
  /// gf.to_node(n_gf);
}

int32 LowOrderField::components() const
{
  return m_values.ncomp();
}

void LowOrderField::to_blueprint(conduit::Node &n_dataset)
{
  conduit::Node &n_field = n_dataset["fields/" + m_name];

  // hard coded topology
  const std::string topo_name = "topo";
  n_field["topology"] = "topo";
  n_field["association"] = m_assoc == Assoc::Vertex ? "vertex" : "element";
  n_field["values"].set_external(m_values.get_host_ptr(), m_values.size());
}

void LowOrderField::eval(const Array<Location> locs, Array<Float> &values)
{
#warning "LowOrderField::eval() does not interpolate cell-centered fields."

  const int32 size = locs.size();
  // allow people to pass in values
  if(values.size() != size)
  {
    values.resize(size);
  }

  const Float * field_vals_ptr = this->values().get_device_ptr_const();
  const Location *locs_ptr = locs.get_device_ptr_const();
  Float * values_ptr = values.get_device_ptr();

  if (this->assoc() == Assoc::Element)  // Assoc::Element --> lookup
  {
    RAJA::forall<for_policy> (RAJA::RangeSegment (0, size), [=] DRAY_LAMBDA (int32 index)
    {
      const Location loc = locs_ptr[index];
      if(loc.m_cell_id != -1)
      {
        values_ptr[index] = field_vals_ptr[loc.m_cell_id];
      }
    });
  }
  else  // Assoc::Vertex  --> interpolate
  {
    // implicit connectivity
    const Vec<int32, 3> dims = this->cell_dims();
    const Vec<int32, 3> vdims = {{dims[0] + 1, dims[1] + 1, dims[2] + 1}};

    RAJA::forall<for_policy> (RAJA::RangeSegment (0, size), [=] DRAY_LAMBDA (int32 index)
    {
      const Location loc = locs_ptr[index];
      if(loc.m_cell_id != -1)
      {
        // 3D cell index
        int32 cell_id = loc.m_cell_id;
        Vec<int32, 3> cell_ijk = {{0, 0, 0}};
        for (int32 d = 0; d < 3; ++d)
        {
          cell_ijk[d] = cell_id % dims[d];
          cell_id /= dims[d];
        }

        // interpolate
        const Vec<Float, 3> &ref_coord = loc.m_ref_pt;
        const Float &rx = ref_coord[0];
        const Float &ry = ref_coord[1];
        const Float &rz = ref_coord[2];
        Float faces[2] = {0, 0};
        for (int32 k = 0; k < 2; ++k)
        {
          Float edges[2] = {0, 0};
          for (int32 j = 0; j < 2; ++j)
          {
            const int32 vert0 = ((cell_ijk[2] + k) * vdims[1]
                                 + (cell_ijk[1] + j)) * vdims[0]
                                 + cell_ijk[0];
            const int32 vert1 = vert0 + 1;

            const Float val0 = field_vals_ptr[vert0];
            const Float val1 = field_vals_ptr[vert1];
            edges[j] = val0 * (1 - ref_coord[0]) + val1 * (ref_coord[0]);
          }
          faces[k] = edges[0] * (1 - ref_coord[1]) + edges[1] * (ref_coord[1]);
        }
        Float value = faces[0] * (1 - ref_coord[2]) + faces[1] * (ref_coord[2]);

        values_ptr[index] = value;
      }
    });
  }
}
} // namespace dray
