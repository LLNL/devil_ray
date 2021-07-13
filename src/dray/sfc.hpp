// copyright 2019 lawrence livermore national security, llc and other
// devil ray developers. see the top-level copyright file for details.
//
// spdx-license-identifier: (bsd-3-clause)


#ifndef DRAY_SFC_HPP
#define DRAY_SFC_HPP

#include <dray/types.hpp>
#include <dray/exports.hpp>

#include <iostream>
#include <ostream>
#include <bitset>

#define HILBERT

namespace dray
{
  /** Traversal state for recursively-defined space-filling-curve. */

  namespace sfc
  {
    typedef uint8 ChildNum;
    typedef uint8 SubIndex;

    struct ReflectRotate
    {
      uint8 m_reflect = 0u;
      uint8 m_permute = 0u | 4u | 32u | 192u;
    };
  }

  //
  // SFC_Morton
  //
  template <int32 dim>
  class SFC_Morton
  {
    public:
      DRAY_EXEC SFC_Morton() = default;
      DRAY_EXEC SFC_Morton(const SFC_Morton &) = default;
      DRAY_EXEC SFC_Morton(SFC_Morton &&) = default;
      DRAY_EXEC SFC_Morton & operator=(const SFC_Morton &) = default;
      DRAY_EXEC SFC_Morton & operator=(SFC_Morton &&) = default;

      // permutation
      DRAY_EXEC sfc::ChildNum child_num(sfc::SubIndex i) const;
      DRAY_EXEC sfc::SubIndex child_rank(sfc::ChildNum cn) const;

      // traversal
      DRAY_EXEC SFC_Morton subcurve(sfc::SubIndex i) const;

      sfc::ReflectRotate orientation() const { return sfc::ReflectRotate{}; }
  };



  //
  // SFC_Hilbert
  //
  template <int32 dim>
  class SFC_Hilbert
  {
    private:
      sfc::ReflectRotate m_orientation;
    public:
      DRAY_EXEC SFC_Hilbert();
      DRAY_EXEC SFC_Hilbert(const SFC_Hilbert &) = default;
      DRAY_EXEC SFC_Hilbert(SFC_Hilbert &&) = default;
      DRAY_EXEC SFC_Hilbert & operator=(const SFC_Hilbert &) = default;
      DRAY_EXEC SFC_Hilbert & operator=(SFC_Hilbert &&) = default;

      // permutation
      DRAY_EXEC sfc::ChildNum child_num(sfc::SubIndex i) const;
      DRAY_EXEC sfc::SubIndex child_rank(sfc::ChildNum cn) const;

      // traversal
      DRAY_EXEC SFC_Hilbert subcurve(sfc::SubIndex i) const;

      sfc::ReflectRotate orientation() const { return m_orientation; }
  };



  // Morton child_num()
  template <int32 dim>
  DRAY_EXEC sfc::ChildNum SFC_Morton<dim>::child_num(sfc::SubIndex i) const
  {
    return i;
  }

  // Morton child_rank()
  template <int32 dim>
  DRAY_EXEC sfc::SubIndex SFC_Morton<dim>::child_rank(sfc::ChildNum cn) const
  {
    return cn;
  }

  // Morton subcurve()
  template <int32 dim>
  DRAY_EXEC SFC_Morton<dim> SFC_Morton<dim>::subcurve(sfc::SubIndex i) const
  {
    return SFC_Morton();
  }


  // Hilbert curve implementation informed by
  // - Haverkort, 2012
  //     "Harmonious Hilbert Curves and Other
  //     Extradimensional Space-filling Curves"
  // - Dendro-KT, gen/RotationTableHilbert.cpp
  //     github.com/paralab/Dendro-KT/blob/master/gen/RotationTableHilbert.cpp

  namespace sfc
  {
    // reflected_gray()
    DRAY_EXEC ChildNum reflected_gray(SubIndex subindex)
    {
      return (subindex >> 1) ^ subindex;
    }

    // inverse_reflected_gray()
    template <int32 dim>
    DRAY_EXEC SubIndex inverse_reflected_gray(ChildNum location)
    {
      SubIndex subindex = 0;
      for (int32 d = 0; d < dim; ++d)
        subindex ^= location >> d;
      return subindex;
    }

    // identity_orientation()
    template <int32 dim>
    DRAY_EXEC ReflectRotate identity_orientation()
    {
      ReflectRotate identity;
      identity.m_reflect = 0;
      identity.m_permute = 0;
      for (int32 d = 0; d < dim; ++d)
        identity.m_permute |= d << (2*d);
      return identity;
    }

    // local_to_world()
    template <int32 dim>
    DRAY_EXEC ChildNum local_to_world(const ReflectRotate &pi, ChildNum cn)
    {
      // Permute axes, then reflect.
      ChildNum mapped = 0;
      for (int32 d = 0; d < dim; ++d)
      {
        const uint8 dest_axis = (pi.m_permute >> (2*d)) & 3u;
        const uint8 bit = (cn >> d) & 1u;
        mapped |= bit << dest_axis;
      }
      mapped ^= pi.m_reflect;
      return mapped;
    }

    // world_to_local()
    template <int32 dim>
    DRAY_EXEC ChildNum world_to_local(const ReflectRotate &pi, ChildNum cn)
    {
      // Anti-reflect, then anti-permute.
      cn ^= pi.m_reflect;
      ChildNum antimapped = 0;
      for (int32 d = 0; d < dim; ++d)
      {
        const uint8 world_axis = (pi.m_permute >> (2*d)) & 3u;
        const uint8 bit = (cn >> world_axis) & 1u;
        antimapped |= bit << d;
      }
      return antimapped;
    }

    // compose()
    template <int32 dim>
    DRAY_EXEC ReflectRotate compose(
        const ReflectRotate &outer, const ReflectRotate &inner)
    {
      // Semidirect product: Conjugate reflection, compose rotation.
      ReflectRotate product;
      product.m_reflect = 0;
      product.m_permute = 0;
      for (int32 d = 0; d < dim; ++d)
      {
        const uint8 outer_dest = (outer.m_permute >> (2*d)) & 3u;
        const uint8 r_bit = (inner.m_reflect >> d) & 1u;
        product.m_reflect |= r_bit << outer_dest;

        const uint8 inner_dest = (inner.m_permute >> (2*d)) & 3u;
        const uint8 compose_dest = (outer.m_permute >> (2*inner_dest)) & 3u;
        product.m_permute |= compose_dest << (2*d);
      }
      product.m_reflect ^= outer.m_reflect;
      return product;
    }




    template <int32 dim>
    std::ostream & print(
        std::ostream &out,
        const ReflectRotate &orientation)
    {
      int32 forward_perm[dim];
      int32 inv_perm[dim];
      for (int32 d = 0; d < dim; ++d)
      {
        int32 inv_value = (orientation.m_permute >> (2*d)) & 3u;
        forward_perm[inv_value] = d;
        inv_perm[d] = inv_value;
      }

      // Permutation
      for (int32 d = dim-1; d >= 0; --d)
        out << forward_perm[d];
      out << "\t";

      // Inverse permutation
      for (int32 d = dim-1 ; d >= 0; --d)
        out << inv_perm[d];
      out << "\t";

      // Reflection
      out << std::bitset<dim>(orientation.m_reflect);
      out << "\t";

      return out;
    }

  }

  // SFC_Hilbert()
  template <int32 dim>
  DRAY_EXEC SFC_Hilbert<dim>::SFC_Hilbert()
  : m_orientation(sfc::identity_orientation<dim>())
  {
  }


  // Hilbert child_num()
  template <int32 dim>
  DRAY_EXEC sfc::ChildNum SFC_Hilbert<dim>::child_num(sfc::SubIndex i) const
  {
    sfc::ChildNum gray = sfc::reflected_gray(i);
    return sfc::local_to_world<dim>(m_orientation, gray);
  }

  // Hilbert child_rank()
  template <int32 dim>
  DRAY_EXEC sfc::SubIndex SFC_Hilbert<dim>::child_rank(sfc::ChildNum cn) const
  {
    sfc::ChildNum gray = sfc::world_to_local<dim>(m_orientation, cn);
    return sfc::inverse_reflected_gray<dim>(gray);
  }

  // Hilbert subcurve()
  template <int32 dim>
  DRAY_EXEC SFC_Hilbert<dim> SFC_Hilbert<dim>::subcurve(sfc::SubIndex i) const
  {
    sfc::ReflectRotate child_rotation;

    // reflection
    const uint8 gray = sfc::reflected_gray(i);
    child_rotation.m_reflect = gray;
    if (i > 0)
    {
      const uint8 gray_prev = sfc::reflected_gray(i-1);
      child_rotation.m_reflect = (gray_prev & (~1u)) | ((~gray) & 1u);
    }

    // rotation
    child_rotation.m_permute = 0;
    const uint8 parity = (i & 1u);
    for (int32 d = 0; d < dim; ++d)
    {
      if (((i >> d) & 1u) != parity)
      {
        child_rotation.m_permute <<= 2;
        child_rotation.m_permute |= d;
      }
    }
    for (int32 d = 0; d < dim; ++d)
    {
      if (((i >> d) & 1u) == parity)
      {
        child_rotation.m_permute <<= 2;
        child_rotation.m_permute |= d;
      }
    }

    SFC_Hilbert subcurve;
    subcurve.m_orientation = sfc::compose<dim>(m_orientation, child_rotation);
    return subcurve;
  }


#ifdef HILBERT
  template <int32 dim>
  using SFC = SFC_Hilbert<dim>;
#else
  template <int32 dim>
  using SFC = SFC_Morton<dim>;
#endif

}

#endif//DRAY_SFC_HPP
