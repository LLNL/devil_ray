#ifndef DRAY_BINOMIAL_HPP
#define DRAY_BINOMIAL_HPP

namespace dray
{

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

template <typename T, int32 n>
class BinomRow
{
  static detail::BinomRowInternal<T,n,n> row;

public:
  static const T *get() { return (T *) &row; }
};

template <typename T, int32 n>
detail::BinomRowInternal<T,n,n> BinomRow<T,n>::row;

} // namespace dray

#endif
