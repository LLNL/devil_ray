#ifndef DRAY_BINOMIAL_HPP
#define DRAY_BINOMIAL_HPP

#include <dray/array.hpp>
#include <dray/exports.hpp>

#include <assert.h>

namespace dray
{

// A table of binomial coefficients.
// Upon size_at_least() to more than the current row number,
// this will expand to twice the current row number.
//class BinomTable
struct BinomTable  //DEBUG
{
  int32 m_current_n;
  Array<int32> m_rows;

private:
  BinomTable() { assert(false); };

public:
  BinomTable(int32 N)
  {
    m_current_n = 0; 
    size_at_least(N);
  }

  const int32 *get_host_ptr_const() const { return m_rows.get_host_ptr_const(); }
  const int32 *get_device_ptr_const() const { return m_rows.get_device_ptr_const(); }

  int32 get_current_n() const { return m_current_n; }

  // Returns whether resize took place.
  bool size_at_least(int32 N);   // N starts from 0.

  DRAY_EXEC
  static int32 get(const int32 *row_ptr, int32 N, int32 k)
  { return get_row(row_ptr, N)[k]; }

  DRAY_EXEC
  static const int32 *get_row(const int32 *row_ptr, int32 N)
  { return row_ptr + (N*(N+1)/2); }

  DRAY_EXEC
  static int32 *get_row(int32 *row_ptr, int32 N)
  { return row_ptr + (N*(N+1)/2); }

  // Does not require any global state or precomputed values.
  DRAY_EXEC
  static void fill_single_row(int32 N, int32 *dest);
};

DRAY_EXEC
void
BinomTable::fill_single_row(int32 N, int32 *dest)
{
  // Fill ends with 1.
  dest[0] = dest[N] = 1;

  // Fill middle entries sequentially based on entries to the left.
  // Use rule C(n,k) = (n-k+1)/k C(n,k-1)
  int32 prev = 1;
  for (int32 k = 1, nmkp1 = N;  k <= N/2;  k++, nmkp1--)
  {
    // Integer division with less chance of overflow. (wikipedia.org)
    dest[k] = dest[N-k] = prev = (prev / k) * nmkp1 + (prev % k) * nmkp1 / k;
  }
}

extern BinomTable GlobBinomTable;





//--- Template Meta Programming stuff ---//

// Recursive binomial coefficient template.
template <int32 n, int32 k>
struct Binom
{
  enum { val = Binom<n-1,k-1>::val + Binom<n-1,k>::val };
};

// Base cases.
template <int32 n> struct Binom<n,n> { enum { val = 1 }; };
template <int32 n> struct Binom<n,0> { enum { val = 1 }; };

namespace detail
{
  // Hack: Inherited data members are layed out contiguously.
  // Use inheritance to store constexpr Binom<> values one at a time recursively.
  template <typename T, int32 n, int32 k>
  struct BinomRowInternal : public BinomRowInternal<T,n,k-1>
  {
    const T cell = static_cast<T>(Binom<n,k>::val);
  };
  
  // Base case.
  template <typename T, int32 n>
  struct BinomRowInternal<T,n,0> { const T cell = static_cast<T>(1); };
}

// - To get a pointer to binomial coefficients stored in a static member:
//   const T *binomial_array = BinomRow<T,n>::get_static();
//
// - To store binomial coefficient literals in a local array variable,
//   1.    BinomRow<T,n> local_row;
//   2.    const T *local_binomial_array = local_row.get();
template <typename T, int32 n>
class BinomRow
{
  static detail::BinomRowInternal<T,n,n> row_static;  // Literals are fed into static member.
  detail::BinomRowInternal<T,n,n> m_row;              // Literals are fed into local memory.

public:
  static const T *get_static() { return (T *) &row_static; }

  DRAY_EXEC
  const T *get() { return (T *) &m_row; }

};

template <typename T, int32 n>
detail::BinomRowInternal<T,n,n> BinomRow<T,n>::row_static;

} // namespace dray

#endif
