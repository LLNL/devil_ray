// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <dray/filters/isosurfacing.hpp>
#include <dray/error.hpp>
#include <dray/dispatcher.hpp>

#include <RAJA/RAJA.hpp>
#include <dray/policies.hpp>
#include <dray/exports.hpp>

#include <dray/derived_topology.hpp>
#include <dray/GridFunction/field.hpp>
#include <dray/GridFunction/device_field.hpp>
#include <dray/Element/elem_attr.hpp>
#include <dray/Element/iso_ops.hpp>

// ----------------------------------------------------
// Isosurfacing approach based on
//   https://dx.doi.org/10.1016/j.cma.2016.10.019
//
// @article{FRIES2017759,
//   title = "Higher-order meshing of implicit geometries—Part I: Integration and interpolation in cut elements",
//   journal = "Computer Methods in Applied Mechanics and Engineering",
//   volume = "313",
//   pages = "759 - 784",
//   year = "2017",
//   issn = "0045-7825",
//   doi = "https://doi.org/10.1016/j.cma.2016.10.019",
//   url = "http://www.sciencedirect.com/science/article/pii/S0045782516308696",
//   author = "T.P. Fries and S. Omerović and D. Schöllhammer and J. Steidl",
// }
// ----------------------------------------------------

namespace dray
{
  // -----------------------
  // Getter/setters
  // -----------------------
  void ExtractIsosurface::iso_field(const std::string field_name)
  {
    m_iso_field_name = field_name;
  }

  std::string ExtractIsosurface::iso_field() const
  {
    return m_iso_field_name;
  }

  void ExtractIsosurface::iso_value(const float32 iso_value)
  {
    m_iso_value = iso_value;
  }

  Float ExtractIsosurface::iso_value() const
  {
    return m_iso_value;
  }
  // -----------------------


  //
  // execute(topo, field)
  //
  template <class MElemT, class FElemT>
  DataSet ExtractIsosurface_execute(DerivedTopology<MElemT> &topo,
                                    Field<FElemT> &field,
                                    Float iso_value)
  {
    static_assert(FElemT::get_ncomp() == 1, "Can't take isosurface of a vector field");

    const Float isoval = iso_value;  // Local for capture
    const int32 n_el_in = field.get_num_elem();
    DeviceField<FElemT> dfield(field);

    constexpr int32 subelem_budget = 10;
    Array<uint8> count_subelem_required;
    Array<uint8> budget_maxed;

    RAJA::forall<for_policy>(RAJA::RangeSegment(0, n_el_in), [=] DRAY_LAMBDA (int32 i) {
        using eops::measure_isocut;
        using eops::IsocutInfo;

        FElemT felem = dfield.get_elem(i);

        const auto shape = adapt_get_shape(felem);
        constexpr ElemType etype = eattr::get_etype(shape);
        const auto order_p = dfield.get_order_policy();
        const int32 p = eattr::get_order(order_p);

        IsocutInfo isocut_info;
        isocut_info = measure_isocut(shape, felem.read_dof_ptr(), isoval, p);
        Split<etype> binary_split = pick_iso_simple_split(shape, isocut_info);

        if (isocut_info.m_cut_type_flag == 0)
          std::cout << "No cut\n";
        else if (isocut_info.m_cut_type_flag & IsocutInfo::CutSimpleTri)
          std::cout << "Simple tri\n";
        else if (isocut_info.m_cut_type_flag & IsocutInfo::CutSimpleQuad)
          std::cout << "Simple quad\n";
        else
          std::cout << "Not simple!  \t " << binary_split << "\n";
    });

    DRAY_ERROR("Implementation of ExtractIsosurface_execute() not done yet");
  }


  // ExtractIsosurfaceFunctor
  struct ExtractIsosurfaceFunctor
  {
    Float m_iso_value;

    DataSet m_output;

    ExtractIsosurfaceFunctor(Float iso_value)
      : m_iso_value{iso_value}
    { }

    template <typename TopologyT, typename FieldT>
    void operator()(TopologyT &topo, FieldT &field)
    {
      m_output = ExtractIsosurface_execute(topo, field, m_iso_value);
    }
  };

  // execute() wrapper
  DataSet ExtractIsosurface::execute(DataSet &data_set)
  {
    ExtractIsosurfaceFunctor func(m_iso_value);
    dispatch_3d(data_set.topology(), data_set.field(m_iso_field_name), func);
    return func.m_output;
  }

}
