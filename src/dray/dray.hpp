// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_HPP
#define DRAY_HPP

namespace dray
{

enum subdivison_strategy_t {fixed = 0, wang, recursive_subdivision};

class dray
{
  public:
  void about ();
  void init ();
  void finalize ();

  static bool cuda_enabled ();

  static void set_face_subdivisions (const int num_subdivions);
  static void set_zone_subdivisions (const int num_subdivions);
  static void set_face_flatness_tolerance(const float tolerance);
  static void set_zone_flatness_tolerance(const float tolerance);
  static void set_face_subdivison_strategy(const subdivison_strategy_t strategy);
  static void set_zone_subdivison_strategy(const subdivison_strategy_t strategy);

  static int get_face_subdivisions ();
  static int get_zone_subdivisions ();
  static float get_face_flatness_tolerance();
  static float get_zone_flatness_tolerance();
  static subdivison_strategy_t get_face_subdivison_strategy();
  static subdivison_strategy_t get_zone_subdivison_strategy();

  private:
  static int m_face_subdivisions;
  static int m_zone_subdivisions;
  static float m_face_flatness_tolerance;
  static float m_zone_flatness_tolerance;
  static subdivison_strategy_t m_face_subdivison_strategy;
  static subdivison_strategy_t m_zone_subdivison_strategy;
};

} // namespace dray
#endif
