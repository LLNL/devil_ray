// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_RAISE_DEGREE_HPP
#define DRAY_RAISE_DEGREE_HPP

#include <dray/data_set.hpp>
#include <dray/GridFunction/mesh.hpp>
#include <dray/Element/element.hpp>

namespace dray
{

class RaiseDegreeDG
{
protected:
public:
  /**
   * @brief Raises the degree of the topology (shape data) in a dataset.
   * @param data_set The dataset whose degree is to be raised. Probably very curved.
   * @param raise The amount to raise the degree by. Supported values are 1, 2, and 3.
   * @return A new dataset with the same field(s) data, but the physical shape
   *         of elements is represented using a higher degree basis for tighter bounds.
   * @note RaiseDegreeDG returns elements that no longer share degrees of freedom.
   */
  DataSet execute(DataSet &data_set, uint32 raise);

  template<class ElemT>
  DataSet execute(Mesh<ElemT> &mesh, DataSet &data_set, uint32 raise);

  template<class ElemT, uint32 raise>
  DataSet execute(Mesh<ElemT> &mesh_lo, DataSet &data_set);
};

};//namespace dray

#endif//DRAY_RAISE_DEGREE_HPP
