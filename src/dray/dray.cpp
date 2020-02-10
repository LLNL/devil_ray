// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <dray/dray.hpp>
#include <dray/exports.hpp>
#include <iostream>

namespace dray
{

int dray::m_face_subdivisions = 1;
int dray::m_zone_subdivisions = 1;
float dray::m_face_flatness_tolerance = 1.0;
float dray::m_zone_flatness_tolerance = 1.0;
subdivison_strategy_t dray::m_face_subdivison_strategy = fixed;
subdivison_strategy_t dray::m_zone_subdivison_strategy = wang;

void dray::set_face_subdivisions (int num_subdivisions)
{
  m_face_subdivisions = num_subdivisions;
}

void dray::set_zone_subdivisions (int num_subdivisions)
{
  m_zone_subdivisions = num_subdivisions;
}

void dray::set_face_flatness_tolerance(float tolerance)
{
  m_face_flatness_tolerance  = tolerance;
}

void dray::set_zone_flatness_tolerance(float tolerance)
{
  m_zone_flatness_tolerance  = tolerance;
}

void dray::set_face_subdivison_strategy(const subdivison_strategy_t strategy) {
  m_face_subdivison_strategy = strategy;
}

void dray::set_zone_subdivison_strategy(const subdivison_strategy_t strategy) {
  m_zone_subdivison_strategy = strategy;
}

int dray::get_zone_subdivisions ()
{
  return m_zone_subdivisions;
}

int dray::get_face_subdivisions ()
{
  return m_zone_subdivisions;
}

float dray::get_face_flatness_tolerance()
{
  return m_face_flatness_tolerance;
}

float dray::get_zone_flatness_tolerance()
{
  return m_zone_flatness_tolerance;
}

subdivison_strategy_t dray::get_face_subdivison_strategy() 
{
  return m_face_subdivison_strategy;
}
subdivison_strategy_t dray::get_zone_subdivison_strategy()
{
  return m_zone_subdivison_strategy;
}

void dray::init ()
{
}

void dray::finalize ()
{
}

bool dray::cuda_enabled ()
{
#ifdef DRAY_CUDA_ENABLED
  return true;
#else
  return false;
#endif
}

void dray::about ()
{
  std::cout << "                                          v0.0.1               "
               "                       \n\n\n";
  std::cout << "                                       @          &,           "
               "                           \n";
  std::cout << "                                      @&          .@           "
               "                           \n";
  std::cout << "                                      @%          .@*          "
               "                           \n";
  std::cout << "                                    &@@@@@@@@@@@@@@@@@/        "
               "                           \n";
  std::cout << "                                   @@@@@@@@@@@@@@@@@@@@.       "
               "                           \n";
  std::cout << "                                   @@@@@@@@@@@@@@@@@@@@,       "
               "                           \n";
  std::cout << "                                 /@@@@@@@@@@@@@@@@@@@@@@@      "
               "                           \n";
  std::cout << "                               &@@@@@@@@@@@@@@@@@@@@@@@@@@@    "
               "                           \n";
  std::cout
  << "                         ,%&@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@%*        "
     "                 \n";
  std::cout
  << "                   (@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@&  "
     "                 \n";
  std::cout << "              "
               ",@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@# "
               "             \n";
  std::cout << "           "
               "(@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
               "@@@@@.          \n";
  std::cout << "        "
               "/@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
               "@@@@@@@@@@&        \n";
  std::cout << "      "
               "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
               "@@@@@@@@@@@@@@@.     \n";
  std::cout << "   "
               ".@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
               "@@@@@@@@@@@@@@@@@@@@%   \n";
  std::cout << " .@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
               "@@@@@@@@@@@@@@@@@@@@@@@@@# \n";
  std::cout
  << "@#                 /@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@&  "
     "               *@\n";
  std::cout
  << "                        @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@*      "
     "                 \n";
  std::cout
  << "                           ,@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@          "
     "                 \n";
  std::cout << "                              &@@@@@@@@@@@@@@@@@@@@@@@@@@@@@   "
               "                           \n";
  std::cout << "                                 &@@@@@@@@@@@@@@@@@@@@@@@      "
               "                           \n";
  std::cout << "                                    #@@@@@@@@@@@@@@@@@.        "
               "                           \n";
  std::cout << "                                       .@@@@@@@@@@#            "
               "                           \n";
  std::cout << "                                           @@@@.               "
               "                           \n";
  std::cout << "                                            &@%                "
               "                           \n";
  std::cout << "                                            ,@(                "
               "                           \n";
  std::cout << "                                             @,                "
               "                           \n";
  std::cout << "                                             @                 "
               "                           \n";
  std::cout << "                                           ,,@,,               "
               "                           \n";
  std::cout << "                                      /@&.*  @  *,@@*          "
               "                           \n";
  std::cout << "                                    %@       @       &&        "
               "                           \n";
  std::cout << "                                   *&        @        %.       "
               "                           \n";
  std::cout << "                                    @        @        @        "
               "                           \n";
  std::cout << "                                     @       @       @.        "
               "                           \n";
  std::cout << "                                      @      @      @.         "
               "                           \n";
  std::cout << "                                      @      @      @          "
               "                           \n";
  std::cout << "                                      @      @      @          "
               "                           \n";
  std::cout << "                                   &@@       @       @@/       "
               "                           \n";
  std::cout << "                                   @@.       @        @&       "
               "                           \n";
  std::cout << "                                           (@@@&               "
               "                           \n";
  std::cout << "                                            *@%                "
               "                           \n";
}

} // namespace dray
