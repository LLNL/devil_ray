// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_ERROR_HPP
#define DRAY_ERROR_HPP

#include <exception>
#include <string>

namespace dray
{

class DRayError : public std::exception
{
  private:
  std::string m_message;
  DRayError ()
  {
  }

  public:
  DRayError (const std::string message) : m_message (message)
  {
  }
  const std::string &GetMessage () const
  {
    return this->m_message;
  }
  const char *what () const noexcept override
  {
    return m_message.c_str ();
  }
};

} // namespace dray
#endif
