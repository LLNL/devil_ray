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
  HexRecord::HexRecord(const std::map<std::string, int32> &coord_idx,
                       const std::map<std::string, int32> &scalar_vidx,
                       const std::map<std::string, int32> &scalar_eidx)
    : HexRecord(coord_idx, scalar_vidx, scalar_eidx, {}, {})
  {
  }

  HexRecord::HexRecord(const std::map<std::string, int32> &coord_idx,
                       const std::map<std::string, int32> &scalar_vidx,
                       const std::map<std::string, int32> &scalar_eidx,
                       const std::map<std::string, int32> &vector_vidx,
                       const std::map<std::string, int32> &vector_eidx )
    : m_birthtime(0),
      m_is_immortal(false),

      m_coord_idx(coord_idx),
      m_coord_data_initd(coord_idx.size(), false),
      m_coord_data(coord_idx.size()),
      m_coord_name(coord_idx.size()),

      m_scalar_vidx(scalar_vidx),
      m_scalar_vdata_initd(scalar_vidx.size(), false),
      m_scalar_vdata(scalar_vidx.size()),
      m_scalar_vname(scalar_vidx.size()),
      m_scalar_eidx(scalar_eidx),
      m_scalar_edata_initd(scalar_eidx.size(), false),
      m_scalar_edata(scalar_eidx.size()),
      m_scalar_ename(scalar_eidx.size()),

      m_vector_vidx(vector_vidx),
      m_vector_vdata_initd(vector_vidx.size(), false),
      m_vector_vdata(vector_vidx.size()),
      m_vector_vname(vector_vidx.size()),
      m_vector_eidx(vector_eidx),
      m_vector_edata_initd(vector_eidx.size(), false),
      m_vector_edata(vector_eidx.size()),
      m_vector_ename(vector_eidx.size())

  {
    for (const auto &name_idx : coord_idx)
      m_coord_name.at(name_idx.second) = name_idx.first;
    for (const auto &name_idx : scalar_vidx)
      m_scalar_vname.at(name_idx.second) = name_idx.first;
    for (const auto &name_idx : scalar_eidx)
      m_scalar_ename.at(name_idx.second) = name_idx.first;
    for (const auto &name_idx : vector_vidx)
      m_vector_vname.at(name_idx.second) = name_idx.first;
    for (const auto &name_idx : vector_eidx)
      m_vector_ename.at(name_idx.second) = name_idx.first;
  }

  /** is_initd_self() */
  bool HexRecord::is_initd_self() const
  {
    return (   *std::min_element(m_coord_data_initd.begin(),   m_coord_data_initd.end())
            && *std::min_element(m_scalar_vdata_initd.begin(), m_scalar_vdata_initd.end())
            && *std::min_element(m_scalar_edata_initd.begin(), m_scalar_edata_initd.end())
            && *std::min_element(m_vector_vdata_initd.begin(), m_vector_vdata_initd.end())
            && *std::min_element(m_vector_edata_initd.begin(), m_vector_edata_initd.end())
            );
  }

  /** is_initd_extern() */
  bool HexRecord::is_initd_extern(const std::map<std::string, int32> &coord_idx,
                                  const std::map<std::string, int32> &scalar_vidx,
                                  const std::map<std::string, int32> &scalar_eidx,
                                  const std::map<std::string, int32> &vector_vidx,
                                  const std::map<std::string, int32> &vector_eidx ) const
  {
    bool initd = true;
    for (const auto & name_idx : coord_idx)
      initd &= m_coord_data_initd[m_coord_idx.at(name_idx.first)];
    for (const auto & name_idx : scalar_vidx)
      initd &= m_scalar_vdata_initd[m_scalar_vidx.at(name_idx.first)];
    for (const auto & name_idx : scalar_eidx)
      initd &= m_scalar_edata_initd[m_scalar_eidx.at(name_idx.first)];
    for (const auto & name_idx : vector_vidx)
      initd &= m_vector_vdata_initd[m_vector_vidx.at(name_idx.first)];
    for (const auto & name_idx : vector_eidx)
      initd &= m_vector_edata_initd[m_vector_eidx.at(name_idx.first)];
    return initd;
  }

  /** print_uninitd_coords */
  void HexRecord::print_uninitd_coords(bool println) const
  {
    const char *RED = "\u001b[31m";
    const char *NRM = "\u001b[0m";

    const char end = (println ? '\n' : ' ');
    for (int32 idx = 0; idx < m_coord_name.size(); ++idx)
      if (!m_coord_data_initd[idx])
        printf("%sCoordinate data '%s' is uninitialized.%c%s", RED, m_coord_name[idx].c_str(), end, NRM);
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
    for (int32 idx = 0; idx < m_vector_vname.size(); ++idx)
      if (!m_vector_vdata_initd[idx])
        printf("%sField data (vert) '%s' is uninitialized.%c%s", RED, m_vector_vname[idx].c_str(), end, NRM);
    for (int32 idx = 0; idx < m_vector_ename.size(); ++idx)
      if (!m_vector_edata_initd[idx])
        printf("%sField data (elem) '%s' is uninitialized.%c%s", RED, m_vector_ename[idx].c_str(), end, NRM);
  }

  /** reset_all() */
  void HexRecord::reset_all()
  {
    m_coord_data_initd.clear();
    m_coord_data_initd.resize(m_coord_idx.size(), false);
    m_scalar_vdata_initd.clear();
    m_scalar_vdata_initd.resize(m_scalar_vidx.size(), false);
    m_scalar_edata_initd.clear();
    m_scalar_edata_initd.resize(m_scalar_eidx.size(), false);
    m_vector_vdata_initd.clear();
    m_vector_vdata_initd.resize(m_vector_vidx.size(), false);
    m_vector_edata_initd.clear();
    m_vector_edata_initd.resize(m_vector_eidx.size(), false);
  }

  /** reuse_all() */
  void HexRecord::reuse_all()
  {
    m_coord_data_initd.clear();
    m_coord_data_initd.resize(m_coord_idx.size(), true);
    m_scalar_vdata_initd.clear();
    m_scalar_vdata_initd.resize(m_scalar_vidx.size(), true);
    m_scalar_edata_initd.clear();
    m_scalar_edata_initd.resize(m_scalar_eidx.size(), true);
    m_vector_vdata_initd.clear();
    m_vector_vdata_initd.resize(m_vector_vidx.size(), true);
    m_vector_edata_initd.clear();
    m_vector_edata_initd.resize(m_vector_eidx.size(), true);
  }

  /** coord_data() */
  const HexRecord::CoordT & HexRecord::coord_data(const std::string &cname) const
  {
    return m_coord_data[m_coord_idx.at(cname)];
  }

  /** coord_data() */
  void HexRecord::coord_data(const std::string &cname, const CoordT &coord_data)
  {
    const int32 idx = m_coord_idx.at(cname);
    m_coord_data[idx] = coord_data;
    m_coord_data_initd[idx] = true;
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

  /** vector_vdata() */
  const HexRecord::VVectorT & HexRecord::vector_vdata(const std::string &fname) const
  {
    return m_vector_vdata[m_vector_vidx.at(fname)];
  }

  /** vector_vdata() */
  void HexRecord::vector_vdata(const std::string &fname, const VVectorT &vdata)
  {
    const int32 idx = m_vector_vidx.at(fname);
    m_vector_vdata[idx] = vdata;
    m_vector_vdata_initd[idx] = true;
  }

  /** vector_edata() */
  const HexRecord::EVectorT & HexRecord::vector_edata(const std::string &fname) const
  {
    return m_vector_edata[m_vector_eidx.at(fname)];
  }

  /** vector_edata() */
  void HexRecord::vector_edata(const std::string &fname, const EVectorT &edata)
  {
    const int32 idx = m_vector_eidx.at(fname);
    m_vector_edata[idx] = edata;
    m_vector_edata_initd[idx] = true;
  }




  //
  // DataSetBuilder definitions
  //

  /** shape_npe[] */
  int32 DataSetBuilder::shape_npe[DataSetBuilder::NUM_SHAPES] = {8, 4};

  /** DataSetBuilder() */
  DataSetBuilder::DataSetBuilder(ShapeMode shape_mode,
                                 const std::vector<std::string> &coord_names,
                                 const std::vector<std::string> &scalar_vnames,
                                 const std::vector<std::string> &scalar_enames,
                                 const std::vector<std::string> &vector_vnames,
                                 const std::vector<std::string> &vector_enames )
    : m_shape_mode(shape_mode),
      m_num_timesteps(1),
      m_num_elems(0),
      m_num_verts(0),
      m_timesteps(),
      m_is_immortal(),
      m_coord_data(coord_names.size()),
      m_scalar_vdata(scalar_vnames.size()),
      m_scalar_edata(scalar_enames.size()),
      m_vector_vdata(vector_vnames.size()),
      m_vector_edata(vector_enames.size())
  {
    int32 idx;

    idx = 0;
    for (const std::string &cname : coord_names)
      m_coord_idx[cname] = idx++;

    idx = 0;
    for (const std::string &fname : scalar_vnames)
      m_scalar_vidx[fname] = idx++;

    idx = 0;
    for (const std::string &fname : scalar_enames)
      m_scalar_eidx[fname] = idx++;

    idx = 0;
    for (const std::string &fname : vector_vnames)
      m_vector_vidx[fname] = idx++;

    idx = 0;
    for (const std::string &fname : vector_enames)
      m_vector_eidx[fname] = idx++;
  }

  /** new_empty_hex_record() */
  HexRecord DataSetBuilder::new_empty_hex_record() const
  {
    if (m_shape_mode != Hex)
      throw std::logic_error("Cannot call new_empty_hex_record() on a non-Hex DataSetBuilder.");
    return HexRecord(m_coord_idx, m_scalar_vidx, m_scalar_eidx, m_vector_vidx, m_vector_eidx);
  }

  /** add_hex_record() : Copies all registered data fields, then flags them as uninitialized. */
  void DataSetBuilder::add_hex_record(HexRecord &record)
  {
    using VScalarT = HexRecord::VScalarT;
    using EScalarT = HexRecord::EScalarT;
    using VVectorT = HexRecord::VVectorT;
    using EVectorT = HexRecord::EVectorT;
    using CoordT   = HexRecord::CoordT;

    if (m_shape_mode != Hex)
      throw std::logic_error("Cannot call add_hex_record() on a non-Hex DataSetBuilder.");

    if (!record.is_initd_extern(m_coord_idx, m_scalar_vidx, m_scalar_eidx, m_vector_vidx, m_vector_eidx))
    {
      record.print_uninitd_coords();
      record.print_uninitd_fields();
      throw std::logic_error("Attempt to add to DataSetBuilder, but record is missing coords/fields.");
    }

    constexpr int32 verts_per_elem = 8;
    const int32 vtk_2_lex[8] = {0, 1, 3, 2,  4, 5, 7, 6};

    m_timesteps.push_back(record.birthtime());
    m_is_immortal.push_back(record.immortal());

    m_num_timesteps = fmax(m_num_timesteps, record.birthtime() + 1);

    m_num_elems++;
    m_num_verts += verts_per_elem;

    for (const auto &name_idx : m_coord_idx)
    {
      const std::string &cname = name_idx.first;
      const int32 cidx = name_idx.second;
      const CoordT &fdata = record.coord_data(cname);
      for (int32 j = 0; j < verts_per_elem; ++j)
        m_coord_data[cidx].push_back(fdata.m_data[vtk_2_lex[j]]);
    }

    for (const auto &name_idx : m_scalar_vidx)
    {
      const std::string &fname = name_idx.first;
      const int32 fidx = name_idx.second;
      const VScalarT &fdata = record.scalar_vdata(fname);
      for (int32 j = 0; j < verts_per_elem; ++j)
        m_scalar_vdata[fidx].push_back(fdata.m_data[vtk_2_lex[j]]);
    }

    for (const auto &name_idx : m_scalar_eidx)
    {
      const std::string &fname = name_idx.first;
      const int32 fidx = name_idx.second;
      const EScalarT &fdata = record.scalar_edata(fname);
      m_scalar_edata[fidx].push_back(fdata.m_data[0]);
    }

    for (const auto &name_idx : m_vector_vidx)
    {
      const std::string &fname = name_idx.first;
      const int32 fidx = name_idx.second;
      const VVectorT &fdata = record.vector_vdata(fname);
      for (int32 j = 0; j < verts_per_elem; ++j)
        m_vector_vdata[fidx].push_back(fdata.m_data[vtk_2_lex[j]]);
    }

    for (const auto &name_idx : m_vector_eidx)
    {
      const std::string &fname = name_idx.first;
      const int32 fidx = name_idx.second;
      const EVectorT &fdata = record.vector_edata(fname);
      m_vector_edata[fidx].push_back(fdata.m_data[0]);
    }

    record.reset_all();
  }


  /** to_blueprint() */
  void DataSetBuilder::to_blueprint(conduit::Node &bp_dataset, int32 cycle) const
  {
    /*
     * https://llnl-conduit.readthedocs.io/en/latest/blueprint_mesh.html#outputting-meshes-for-visualization
     */

    const int32 n_elems = m_num_elems;
    const int32 npe = shape_npe[m_shape_mode];

    // Index all element records selected by cycle.
    // TODO if this step gets prohibitively expensive,
    //  sort by cycle on-line in add_XXX_record().
    std::vector<int32> sel;
    for (int32 eid = 0; eid < n_elems; ++eid)
      if (m_timesteps[eid] == cycle || (m_is_immortal[eid] && cycle >= m_timesteps[eid]))
        sel.push_back(eid);

    const int32 n_sel_elems = sel.size();
    const int32 n_sel_verts = n_sel_elems * npe;


    //
    // Init node.
    //
    bp_dataset.reset();
    bp_dataset["state/time"] = (float64) cycle;
    bp_dataset["state/cycle"] = (uint64) cycle;

    conduit::Node &coordsets = bp_dataset["coordsets"];
    conduit::Node &topologies = bp_dataset["topologies"];
    conduit::Node &fields = bp_dataset["fields"];

    //
    // Duplicate fields for each coordset.
    //
    for (const auto &name_idx : m_coord_idx)
    {
      const std::string &cname = name_idx.first;
      const int32 cidx = name_idx.second;

      const std::string topo_name = cname;
      const std::string coordset_name = cname + "_coords";

      //
      // Coordset.
      //
      conduit::Node &coordset = coordsets[coordset_name];
      coordset["type"] = "explicit";
      conduit::Node &coord_vals = coordset["values"];
      coordset["values/x"].set(conduit::DataType::float64(n_sel_verts));
      coordset["values/y"].set(conduit::DataType::float64(n_sel_verts));
      coordset["values/z"].set(conduit::DataType::float64(n_sel_verts));
      float64 *x_vals = coordset["values/x"].value();
      float64 *y_vals = coordset["values/y"].value();
      float64 *z_vals = coordset["values/z"].value();
      const std::vector<Vec<Float, 3>> & in_coord_data = m_coord_data[cidx];
      for (int32 eidx = 0; eidx < n_sel_elems; ++eidx)
        for (int32 nidx = 0; nidx < npe; ++nidx)
        {
          x_vals[eidx * npe + nidx] = (float64) in_coord_data[sel[eidx] * npe + nidx][0];
          y_vals[eidx * npe + nidx] = (float64) in_coord_data[sel[eidx] * npe + nidx][1];
          z_vals[eidx * npe + nidx] = (float64) in_coord_data[sel[eidx] * npe + nidx][2];
        }


      //
      // Topology.
      //
      conduit::Node &topo = topologies[topo_name];
      topo["type"] = "unstructured";
      topo["coordset"] = coordset_name;
      topo["elements/shape"] = "hex";
      topo["elements/connectivity"].set(conduit::DataType::int32(n_sel_verts));
      int32 * conn = topo["elements/connectivity"].value();
      std::iota(conn, conn + n_sel_verts, 0);


      const std::string jstr = "_";

      //
      // Fields.
      //
      for (const auto &name_idx : m_scalar_vidx)  // Scalar vertex fields.
      {
        const std::string &fname = name_idx.first;
        const int32 fidx = name_idx.second;
        const std::string field_name = cname + jstr + fname;

        conduit::Node &field = fields[field_name];
        field["association"] = "vertex";
        field["type"] = "scalar";
        field["topology"] = topo_name;
        field["values"].set(conduit::DataType::float64(n_sel_verts));

        float64 *out_vals = field["values"].value();
        const std::vector<Vec<Float, 1>> &in_field_data = m_scalar_vdata[fidx];
        for (int32 eidx = 0; eidx < n_sel_elems; ++eidx)
          for (int32 nidx = 0; nidx < npe; ++nidx)
            out_vals[eidx * npe + nidx] = in_field_data[sel[eidx] * npe + nidx][0];
      }

      for (const auto &name_idx : m_scalar_eidx)  // Scalar element fields.
      {
        const std::string &fname = name_idx.first;
        const int32 fidx = name_idx.second;
        const std::string field_name = cname + jstr + fname;

        conduit::Node &field = fields[field_name];
        field["association"] = "element";
        field["type"] = "scalar";
        field["topology"] = topo_name;
        field["values"].set(conduit::DataType::float64(n_sel_elems));

        float64 *out_vals = field["values"].value();
        const std::vector<Vec<Float, 1>> &in_field_data = m_scalar_edata[fidx];
        for (int32 eidx = 0; eidx < n_sel_elems; ++eidx)
          out_vals[eidx] = in_field_data[sel[eidx]][0];
      }


      const std::string tangent_names[3] = {"u", "v", "w"};

      for (const auto &name_idx : m_vector_vidx)  // Vector vertex fields.
      {
        const std::string &fname = name_idx.first;
        const int32 fidx = name_idx.second;
        const std::string field_name = cname + jstr + fname;

        constexpr int32 ncomp = 3;
        conduit::Node &field = fields[field_name];
        field["association"] = "vertex";
        field["type"] = "vector";
        field["topology"] = topo_name;
        field["values"];

        float64 * out_vals[ncomp];
        for (int32 d = 0; d < ncomp; ++d)
        {
          field["values"][tangent_names[d]].set(conduit::DataType::float64(n_sel_verts));
          out_vals[d] = field["values"][tangent_names[d]].value();
        }

        const std::vector<Vec<Float, 3>> &in_field_data = m_vector_vdata[fidx];
        for (int32 eidx = 0; eidx < n_sel_elems; ++eidx)
          for (int32 nidx = 0; nidx < npe; ++nidx)
            for (int32 d = 0; d < ncomp; ++d)
              out_vals[d][eidx * npe + nidx] = in_field_data[sel[eidx] * npe + nidx][d];
      }

      for (const auto &name_idx : m_vector_eidx)  // Vector element fields.
      {
        const std::string &fname = name_idx.first;
        const int32 fidx = name_idx.second;
        const std::string field_name = cname + jstr + fname;

        constexpr int32 ncomp = 3;
        conduit::Node &field = fields[field_name];
        field["association"] = "element";
        field["type"] = "vector";
        field["topology"] = topo_name;
        field["values"];

        float64 * out_vals[ncomp];
        for (int32 d = 0; d < ncomp; ++d)
        {
          field["values"][tangent_names[d]].set(conduit::DataType::float64(n_sel_elems));
          out_vals[d] = field["values"][tangent_names[d]].value();
        }

        const std::vector<Vec<Float, 3>> &in_field_data = m_vector_edata[fidx];
        for (int32 eidx = 0; eidx < n_sel_elems; ++eidx)
          for (int32 d = 0; d < ncomp; ++d)
            out_vals[d][eidx] = in_field_data[sel[eidx]][d];
      }
      // End all fields.

    }//for coordset
  }

}//namespace dray
