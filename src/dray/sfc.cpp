// copyright 2019 lawrence livermore national security, llc and other
// devil ray developers. see the top-level copyright file for details.
//
// spdx-license-identifier: (bsd-3-clause)


#include <dray/sfc.hpp>

namespace dray
{
#ifdef HILBERT
  template class SFC_Hilbert<2>;
  template class SFC_Hilbert<3>;
#else
  template class SFC_Morton<2>;
  template class SFC_Morton<3>;
#endif
}
