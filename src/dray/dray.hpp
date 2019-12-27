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

  static bool cuda_enabled ();

  static void set_face_subdivisions (const int num_subdivions);
  static void set_zone_subdivisions (const int num_subdivions);

  static int get_face_subdivisions ();
  static int get_zone_subdivisions ();

  private:
  static int m_face_subdivisions;
  static int m_zone_subdivisions;
};


} // namespace dray
#endif
