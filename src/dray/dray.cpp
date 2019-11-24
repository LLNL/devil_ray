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

void dray::set_face_subdivisions (int num_subdivisions)
{
  m_face_subdivisions = num_subdivisions;
}

void dray::set_zone_subdivisions (int num_subdivisions)
{
  m_zone_subdivisions = num_subdivisions;
}

int dray::get_zone_subdivisions ()
{
  return m_zone_subdivisions;
}

int dray::get_face_subdivisions ()
{
  return m_zone_subdivisions;
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
