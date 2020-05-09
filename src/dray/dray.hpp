// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_HPP
#define DRAY_HPP

namespace dray
{

class dray
{
  public:
  void about ();
  void init ();
  void finalize ();

  bool mpi_enabled();
  int  mpi_size();
  int  mpi_rank();
  void mpi_comm(int mpi_comm_id);
  int  mpi_comm();

  static bool cuda_enabled ();

  static void set_face_subdivisions (const int num_subdivions);
  static void set_zone_subdivisions (const int num_subdivions);

  static int get_face_subdivisions ();
  static int get_zone_subdivisions ();

  // attempt to load fast paths
  // if false, default to general order path
  static void prefer_native_order_mesh(bool on);
  static bool prefer_native_order_mesh();
  static void prefer_native_order_field(bool on);
  static bool prefer_native_order_field();

  private:
  static int m_face_subdivisions;
  static int m_zone_subdivisions;
  static bool m_prefer_native_order_mesh;
  static bool m_prefer_native_order_field;
};

} // namespace dray
#endif
