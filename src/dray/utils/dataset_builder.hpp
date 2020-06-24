// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_DATSET_BUILDER_HPP
#define DRAY_DATSET_BUILDER_HPP

#include <dray/types.hpp>
#include <dray/vec.hpp>
#include <dray/Element/elem_attr.hpp>

#include <conduit.hpp>

#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <numeric>



// Sequential builder to emit cells (disconnected from all other cells),
// with associated vertex coordinates, and vertex- or cell-centered field data.
// Then after finished building, can be converted to a blueprint dataset.

// Supports low-order.

namespace dray
{

  template <typename T, int32 ncomp>
  struct HexVData
  {
    Vec<T, ncomp> m_data[8];
  };

  template <typename T, int32 ncomp>
  struct HexEData
  {
    Vec<T, ncomp> m_data[1];
  };

  /*
  template <typename T, int32 ncomp>
  struct TetVData
  {
    Vec<T, ncomp> m_data[4];
  };

  template <typename T, int32 ncomp>
  struct TetEData
  {
    Vec<T, ncomp> m_data[1];
  };
  */


  /** HexRecord */
  class HexRecord
  {
    public:
      using VScalarT = HexVData<Float, 1>;
      using EScalarT = HexEData<Float, 1>;
      using VVectorT = HexVData<Float, 3>;
      using EVectorT = HexEData<Float, 3>;
      using CoordT = HexVData<Float, 3>;

    private:
      int32 m_birthtime;
      bool m_is_immortal;

      std::vector<CoordT> m_coord_data;
      std::vector<bool> m_coord_data_initd;
      std::map<std::string, int32> m_coord_idx;
      std::vector<std::string> m_coord_name;

      std::vector<VScalarT> m_scalar_vdata;
      std::vector<EScalarT> m_scalar_edata;
      std::vector<VVectorT> m_vector_vdata;
      std::vector<EVectorT> m_vector_edata;
      std::vector<bool> m_scalar_vdata_initd;
      std::vector<bool> m_scalar_edata_initd;
      std::vector<bool> m_vector_vdata_initd;
      std::vector<bool> m_vector_edata_initd;

      std::vector<std::string> m_scalar_vname;
      std::vector<std::string> m_scalar_ename;
      std::vector<std::string> m_vector_vname;
      std::vector<std::string> m_vector_ename;

      std::map<std::string, int32> m_scalar_vidx;
      std::map<std::string, int32> m_scalar_eidx;
      std::map<std::string, int32> m_vector_vidx;
      std::map<std::string, int32> m_vector_eidx;

    public:
      /** HexRecord() : Keeps consistent ordering from input. */
      HexRecord(const std::map<std::string, int32> &coord_idx,
                const std::map<std::string, int32> &scalar_vidx,
                const std::map<std::string, int32> &scalar_eidx);

      HexRecord(const std::map<std::string, int32> &coord_idx,
                const std::map<std::string, int32> &scalar_vidx,
                const std::map<std::string, int32> &scalar_eidx,
                const std::map<std::string, int32> &vector_vidx,
                const std::map<std::string, int32> &vector_eidx);

      /** birthtime() */
      int32 birthtime() const { return m_birthtime; }

      /** birthtime() */
      void birthtime(int32 birthtime) { m_birthtime = birthtime; }

      /** immortal() */
      bool immortal() const { return m_is_immortal; }

      /** immortal() */
      void immortal(bool immortal) { m_is_immortal = immortal; }

      /** is_initd_self() */
      bool is_initd_self() const;

      /** is_initd_extern() */
      bool is_initd_extern(const std::map<std::string, int32> &coord_idx,
                           const std::map<std::string, int32> &scalar_vidx,
                           const std::map<std::string, int32> &scalar_eidx,
                           const std::map<std::string, int32> &vector_vidx,
                           const std::map<std::string, int32> &vector_eidx ) const;

      /** print_uninitd_coords */
      void print_uninitd_coords(bool println = true) const;

      /** print_uninitd_fields */
      void print_uninitd_fields(bool println = true) const;

      /** reset_all() */
      void reset_all();

      /** reuse_all() */
      void reuse_all();

      /** coord_data() */
      const CoordT & coord_data(const std::string &cname) const;

      /** coord_data() */
      void coord_data(const std::string &cname, const CoordT &coord_data);

      /** scalar_vdata() */
      const VScalarT & scalar_vdata(const std::string &fname) const;

      /** scalar_vdata() */
      void scalar_vdata(const std::string &fname, const VScalarT &vdata);

      /** scalar_edata() */
      const EScalarT & scalar_edata(const std::string &fname) const;

      /** scalar_edata() */
      void scalar_edata(const std::string &fname, const EScalarT &edata);

      /** vector_vdata() */
      const VVectorT & vector_vdata(const std::string &fname) const;

      /** vector_vdata() */
      void vector_vdata(const std::string &fname, const VVectorT &vdata);

      /** vector_edata() */
      const EVectorT & vector_edata(const std::string &fname) const;

      /** vector_edata() */
      void vector_edata(const std::string &fname, const EVectorT &edata);

  };


  /** DSBBuffer */
  struct DSBBuffer
  {
    DSBBuffer(int32 n_coordsets,
              int32 n_vscalar,
              int32 n_escalar,
              int32 n_vvector,
              int32 n_evector);

    void clear_records();

    int32 m_num_timesteps;
    int32 m_num_elems;
    int32 m_num_verts;

    std::vector<int32> m_timesteps;
    std::vector<bool> m_is_immortal;

    std::vector<std::vector<Vec<Float, 3>>> m_coord_data;

    std::vector<std::vector<Vec<Float, 1>>> m_scalar_vdata;
    std::vector<std::vector<Vec<Float, 1>>> m_scalar_edata;
    std::vector<std::vector<Vec<Float, 3>>> m_vector_vdata;
    std::vector<std::vector<Vec<Float, 3>>> m_vector_edata;
  };


  /** DataSetBuilder */
  class DataSetBuilder
  {
    public:
      enum ShapeMode { Hex, Tet, NUM_SHAPES };

      DataSetBuilder(ShapeMode shape_mode,
                     const std::vector<std::string> &coord_names,
                     const std::vector<std::string> &scalar_vnames,
                     const std::vector<std::string> &scalar_enames,
                     const std::vector<std::string> &vector_vnames,
                     const std::vector<std::string> &vector_enames );

      /** to_blueprint() : Copies cells tagged for cycle into conduit node, returns number of cells. */
      int32 to_blueprint(conduit::Node &bp_dataset, int32 cycle = 0) const;

      int32 num_timesteps() const { return m_central_buffer.m_num_timesteps; }

      ShapeMode shape_mode() const { return m_shape_mode; }
      void shape_mode_hex() { m_shape_mode = Hex; }
      void shape_mode_tet() { m_shape_mode = Tet; }

      int32 num_buffers() const { return m_inflow_buffers.size(); }
      void resize_num_buffers(int32 num_buffers);
      void flush_and_close_all_buffers();
      void clear_buffer(int32 buffer_id) { m_inflow_buffers[buffer_id].clear_records(); }

      HexRecord new_empty_hex_record() const;

      void add_hex_record(int32 buffer_id, HexRecord &record);
      void add_hex_record_direct(HexRecord &record);

      // Maps from coordset name to coordset index.
      const std::map<std::string, int32> &coord_idx() const { return m_coord_idx; }

      // Maps from field name to field index in corresponding field category.
      const std::map<std::string, int32> &scalar_vidx() const { return m_scalar_vidx; }
      const std::map<std::string, int32> &scalar_eidx() const { return m_scalar_eidx; }
      const std::map<std::string, int32> &vector_vidx() const { return m_vector_vidx; }
      const std::map<std::string, int32> &vector_eidx() const { return m_vector_eidx; }

      // Coordset vector.
      const std::vector<Vec<Float, 3>> &coord_data(int32 idx) const;

      // Vectors of field data, by category.
      const std::vector<Vec<Float, 1>> &scalar_vdata(int32 idx) const;
      const std::vector<Vec<Float, 1>> &scalar_edata(int32 idx) const;
      const std::vector<Vec<Float, 3>> &vector_vdata(int32 idx) const;
      const std::vector<Vec<Float, 3>> &vector_edata(int32 idx) const;

    private:
      static int32 shape_npe[NUM_SHAPES];

      void add_hex_record(DSBBuffer &buffer, HexRecord &record);

      ShapeMode m_shape_mode;

      std::map<std::string, int32> m_coord_idx;
      std::map<std::string, int32> m_scalar_vidx;
      std::map<std::string, int32> m_scalar_eidx;
      std::map<std::string, int32> m_vector_vidx;
      std::map<std::string, int32> m_vector_eidx;

      DSBBuffer m_central_buffer;
      std::vector<DSBBuffer> m_inflow_buffers;
  };

}//namespace dray


#endif//DRAY_DATSET_BUILDER_HPP
