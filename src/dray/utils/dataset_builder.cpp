// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <dray/utils/dataset_builder.hpp>


namespace dray
{
  //
  // HexRecord definitions
  //

  /** HexRecord() : Keeps consistent ordering from input. */
  HexRecord::HexRecord(const std::map<std::string, int32> &scalar_vidx,
                       const std::map<std::string, int32> &scalar_eidx)
    : m_coord_data_initd(false),
      m_coord_data(),
      m_scalar_vidx(scalar_vidx),
      m_scalar_vdata_initd(scalar_vidx.size(), false),
      m_scalar_vdata(scalar_vidx.size()),
      m_scalar_vname(scalar_vidx.size()),
      m_scalar_eidx(scalar_eidx),
      m_scalar_edata_initd(scalar_eidx.size(), false),
      m_scalar_edata(scalar_eidx.size()),
      m_scalar_ename(scalar_eidx.size())
  {
    for (const auto &name_idx : scalar_vidx)
      m_scalar_vname[name_idx.second] = name_idx.first;
    for (const auto &name_idx : scalar_eidx)
      m_scalar_ename[name_idx.second] = name_idx.first;
  }

  /** is_initd_self() */
  bool HexRecord::is_initd_self() const
  {
    return (m_coord_data_initd
            && *std::min_element(m_scalar_vdata_initd.begin(), m_scalar_vdata_initd.end())
            && *std::min_element(m_scalar_edata_initd.begin(), m_scalar_edata_initd.end()));
  }

  /** is_initd_extern() */
  bool HexRecord::is_initd_extern(const std::map<std::string, int32> &scalar_vidx,
                                  const std::map<std::string, int32> &scalar_eidx) const
  {
    bool initd = true;
    initd &= m_coord_data_initd;
    for (const auto & name_idx : scalar_vidx)
      initd &= m_scalar_vdata_initd[m_scalar_vidx.at(name_idx.first)];
    for (const auto & name_idx : scalar_eidx)
      initd &= m_scalar_edata_initd[m_scalar_eidx.at(name_idx.first)];
    return initd;
  }

  /** print_uninitd_coords */
  void HexRecord::print_uninitd_coords(bool println) const
  {
    const char *RED = "\u001b[31m";
    const char *NRM = "\u001b[0m";

    if (!m_coord_data_initd)
      printf("%sCoordinate data is uninitialized.%c%s", RED, (println ? '\n' : ' '), NRM);
  }

  /** print_uninitd_fields */
  void HexRecord::print_uninitd_fields(bool println) const
  {
    const char *RED = "\u001b[31m";
    const char *NRM = "\u001b[0m";

    const char end = (println ? '\n' : ' ');
    for (int32 idx = 0; idx < m_scalar_vname.size(); ++idx)
      if (!m_scalar_vdata_initd[idx])
        printf("%sField data (vert) '%s' is uninitialized.%c%s", RED, m_scalar_vname[idx].c_str(), end, NRM);
    for (int32 idx = 0; idx < m_scalar_ename.size(); ++idx)
      if (!m_scalar_edata_initd[idx])
        printf("%sField data (elem) '%s' is uninitialized.%c%s", RED, m_scalar_ename[idx].c_str(), end, NRM);
  }

  /** reset_extern() */
  void HexRecord::reset_extern(const std::map<std::string, int32> &scalar_vidx,
                               const std::map<std::string, int32> &scalar_eidx)
  {
    m_coord_data_initd = false;
    for (const auto & name_idx : scalar_vidx)
      m_scalar_vdata_initd[m_scalar_vidx.at(name_idx.first)] = false;
    for (const auto & name_idx : scalar_eidx)
      m_scalar_edata_initd[m_scalar_eidx.at(name_idx.first)] = false;
  }

  /** reset_all() */
  void HexRecord::reset_all()
  {
    m_coord_data_initd = false;
    m_scalar_vdata_initd.clear();
    m_scalar_vdata_initd.resize(m_scalar_vidx.size(), false);
    m_scalar_edata_initd.clear();
    m_scalar_edata_initd.resize(m_scalar_eidx.size(), false);
  }

  /** reuse_all() */
  void HexRecord::reuse_all()
  {
    m_coord_data_initd = true;
    m_scalar_vdata_initd.clear();
    m_scalar_vdata_initd.resize(m_scalar_vidx.size(), true);
    m_scalar_edata_initd.clear();
    m_scalar_edata_initd.resize(m_scalar_eidx.size(), true);
  }

  /** coords() */
  const HexRecord::CoordT & HexRecord::coords() const
  {
    return m_coord_data;
  }

  /** coords() */
  void HexRecord::coords(const CoordT &coord_data)
  {
    m_coord_data = coord_data;
    m_coord_data_initd = true;
  }

  /** scalar_vdata() */
  const HexRecord::VScalarT & HexRecord::scalar_vdata(const std::string &fname) const
  {
    return m_scalar_vdata[m_scalar_vidx.at(fname)];
  }

  /** scalar_vdata() */
  void HexRecord::scalar_vdata(const std::string &fname, const VScalarT &vdata)
  {
    const int32 idx = m_scalar_vidx.at(fname);
    m_scalar_vdata[idx] = vdata; 
    m_scalar_vdata_initd[idx] = true;
  }

  /** scalar_edata() */
  const HexRecord::EScalarT & HexRecord::scalar_edata(const std::string &fname) const
  {
    return m_scalar_edata[m_scalar_eidx.at(fname)];
  }

  /** scalar_edata() */
  void HexRecord::scalar_edata(const std::string &fname, const EScalarT &edata)
  {
    const int32 idx = m_scalar_eidx.at(fname);
    m_scalar_edata[idx] = edata;
    m_scalar_edata_initd[idx] = true;
  }




  //
  // DataSetBuilder definitions
  //

  /** DataSetBuilder() */
  DataSetBuilder::DataSetBuilder(ShapeMode shape_mode,
                                 const std::vector<std::string> &scalar_vnames,
                                 const std::vector<std::string> &scalar_enames)
    : m_shape_mode(shape_mode),
      m_num_elems(0),
      m_coord_data(),
      m_scalar_vdata(scalar_vnames.size()),
      m_scalar_edata(scalar_enames.size())
  {
    int32 idx;

    idx = 0;
    for (const std::string &fname : scalar_vnames)
      m_scalar_vidx[fname] = idx++;

    idx = 0;
    for (const std::string &fname : scalar_enames)
      m_scalar_eidx[fname] = idx++;
  }

  /** new_empty_hex_record() */
  HexRecord DataSetBuilder::new_empty_hex_record() const
  {
    if (m_shape_mode != Hex)
      throw std::logic_error("Cannot call new_empty_hex_record() on a non-Hex DataSetBuilder.");
    return HexRecord(m_scalar_vidx, m_scalar_eidx);
  }

  /** add_hex_record() : Copies all registered data fields, then flags them as uninitialized. */
  void DataSetBuilder::add_hex_record(HexRecord &record)
  {
    if (m_shape_mode != Hex)
      throw std::logic_error("Cannot call add_hex_record() on a non-Hex DataSetBuilder.");

    if (!record.is_initd_extern(m_scalar_vidx, m_scalar_eidx))
    {
      record.print_uninitd_coords();
      record.print_uninitd_fields();
      throw std::logic_error("Attempt to add to DataSetBuilder, but record is missing fields.");
    }

    constexpr int32 verts_per_elem = 8;

    m_num_elems++;

    const HexVData<Float, 3> &cdata = record.coords();
    for (int32 j = 0; j < verts_per_elem; ++j)
      m_coord_data.push_back(cdata.m_data[j]);

    for (const auto &name_idx : m_scalar_vidx)
    {
      const std::string &fname = name_idx.first;
      const int32 fidx = name_idx.second;
      const HexVData<Float, 1> &fdata = record.scalar_vdata(fname);
      for (int32 j = 0; j < verts_per_elem; ++j)
        m_scalar_vdata[fidx].push_back(fdata.m_data[j]);
    }

    for (const auto &name_idx : m_scalar_eidx)
    {
      const std::string &fname = name_idx.first;
      const int32 fidx = name_idx.second;
      const HexEData<Float, 1> &fdata = record.scalar_edata(fname);
      m_scalar_edata[fidx].push_back(fdata.m_data[0]);
    }

    record.reset_extern(m_scalar_vidx, m_scalar_eidx);
  }


  /** to_blueprint() */
  void DataSetBuilder::to_blueprint(conduit::Node &bp_dataset,
                                    const std::string &coordset_name) const
  {
    /*
     * https://llnl-conduit.readthedocs.io/en/latest/blueprint_mesh.html#outputting-meshes-for-visualization
     */

    const int32 n_elems = m_num_elems;
    const int32 n_verts = m_coord_data.size();

    //
    // Init node.
    //
    bp_dataset.reset();
    bp_dataset["state/time"] = (float64) 0.0f;
    bp_dataset["state/cycle"] = (uint64) 0;

    conduit::Node &coordset = bp_dataset["coordsets/" + coordset_name];
    conduit::Node &topo = bp_dataset["topologies/mesh"];
    conduit::Node &fields = bp_dataset["fields"];

    //
    // Coordset.
    //
    coordset["type"] = "explicit";
    conduit::Node &coord_vals = coordset["values"];
    coordset["values/x"].set(conduit::DataType::float64(n_verts));
    coordset["values/y"].set(conduit::DataType::float64(n_verts));
    coordset["values/z"].set(conduit::DataType::float64(n_verts));
    float64 *x_vals = coordset["values/x"].value();
    float64 *y_vals = coordset["values/y"].value();
    float64 *z_vals = coordset["values/z"].value();
    for (int32 vidx = 0; vidx < n_verts; ++vidx)
    {
      x_vals[vidx] = (float64) m_coord_data[vidx][0];
      y_vals[vidx] = (float64) m_coord_data[vidx][1];
      z_vals[vidx] = (float64) m_coord_data[vidx][2];
    }

    //
    // Topology.
    //
    topo["type"] = "unstructured";
    topo["coordset"] = coordset_name;
    topo["elements/shape"] = "hex";
    topo["elements/connectivity"].set(conduit::DataType::int32(n_verts));
    int32 * conn = topo["elements/connectivity"].value();
    std::iota(conn, conn + n_verts, 0);

    //
    // Fields.
    //
    for (const auto &name_idx : m_scalar_vidx)  // Scalar vertex fields.
    {
      const std::string &fname = name_idx.first;
      const int32 fidx = name_idx.second;

      constexpr int32 ncomp = 1;
      conduit::Node &field = fields[fname];
      field["association"] = "vertex";
      field["type"] = "scalar";
      field["topology"] = "mesh";
      field["values"].set(conduit::DataType::float64(ncomp * n_verts));

      float64 *out_vals = field["values"].value();
      const std::vector<Vec<Float, 1>> &in_field_data = m_scalar_vdata[fidx];
      for (int32 i = 0; i < in_field_data.size(); ++i)
        out_vals[i] = in_field_data[i][0];
    }

    for (const auto &name_idx : m_scalar_eidx)  // Scalar element fields.
    {
      const std::string &fname = name_idx.first;
      const int32 fidx = name_idx.second;

      constexpr int32 ncomp = 1;
      conduit::Node &field = fields[fname];
      field["association"] = "element";
      field["type"] = "scalar";
      field["topology"] = "mesh";
      field["values"].set(conduit::DataType::float64(ncomp * n_elems));

      float64 *out_vals = field["values"].value();
      const std::vector<Vec<Float, 1>> &in_field_data = m_scalar_edata[fidx];
      for (int32 i = 0; i < in_field_data.size(); ++i)
        out_vals[i] = in_field_data[i][0];
    }

  }

}//namespace dray
