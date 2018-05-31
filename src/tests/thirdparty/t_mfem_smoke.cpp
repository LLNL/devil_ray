#include "gtest/gtest.h"
#include <mfem.hpp>

TEST(mfem_smoke, mfem_smoke)
{
  mfem::FiniteElementCollection *fec;
  int order = 1;
  int dims = 2;
  fec = new mfem::H1_FECollection(order, dims);
  delete fec;
}
