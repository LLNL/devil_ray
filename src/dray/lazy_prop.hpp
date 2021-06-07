// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef DRAY_LAZY_PROP_HPP
#define DRAY_LAZY_PROP_HPP

namespace dray
{

/**
 * @class LazyProp
 * @brief Container that waits to evaluate a property until it is needed.
 *
 * The constructor takes a functor and an argument later passed to the functor.
 * The functor is evaluated upon first invocation of either get() or the
 * conversion operator PropT(). The result is stored in a member (m_data),
 * which is returned directly upon future invocations of get() or PropT(),
 * and is never modified again.
 *
 * What this class accomplishes:
 *
 *   - Objects with a lazy read-only property can retain const,
 *     even if the property must be accessed.
 *
 *   - The internal "mutability" of a lazy property is abstracted from
 *     class member const functions, thus preventing accidental writes.
 */
template <typename PropT, typename Calculator, typename ArgT>
class LazyProp
{
  mutable PropT m_data;
  mutable bool m_calculated;
  const Calculator m_calculator;
  const ArgT m_arg;

  public:
    LazyProp() = default;
    LazyProp(const LazyProp &) = delete;  // Old arg (eg ptr) may not be valid.
    LazyProp(LazyProp &&) = delete;       // Old arg (eg ptr) may not be valid.

    LazyProp(Calculator calculator, ArgT arg)
      : m_calculated(false),
        m_calculator(calculator),
        m_arg(arg)
    {}

    operator const PropT &() const { return this->get(); }

    const PropT & get() const
    {
      if (!m_calculated)
      {
        m_data = m_calculator(m_arg);
        m_calculated = true;
      }
      return m_data;
    }

    void reset()  // force recalculation upon next read
    {
      m_calculated = false;
    }
};

}

#endif//DRAY_LAZY_PROP_HPP
