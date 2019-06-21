#include "gtest/gtest.h"

#include "t_utils.hpp"

#include "dray/GridFunction/mesh.hpp"
#include "dray/aabb.hpp"
#include "dray/range.hpp"
#include "dray/math.hpp"
#include "dray/subdivision_search.hpp"
#include "dray/bernstein_basis.hpp"

/*
 * Tests involving sub-elements and subdivision search.
 */


TEST(dray_subdivision, dray_subelement)
{
  /*TODO*/
}

template <int32 S>
struct RefBox
{
  dray::Range<> m_ranges[S];

  DRAY_EXEC const dray::Range<> & operator[] (dray::int32 ii) const { return m_ranges[ii]; }
  DRAY_EXEC       dray::Range<> & operator[] (dray::int32 ii)       { return m_ranges[ii]; }

  DRAY_EXEC       dray::Range<> * begin()       { return &m_ranges[0]; }
  DRAY_EXEC const dray::Range<> * begin() const { return &m_ranges[0]; }

  template <typename T>
  DRAY_EXEC dray::Vec<T, S> center() const
  {
    dray::Vec<T, S> ret;
    for (int32 d = 0; d < S; d++)
      ret[d] = m_ranges[d].center();
    return ret;
  }
};


TEST(dray_subdivision, dray_subdiv_search)
{
  constexpr dray::int32 p_order = 2;
  constexpr dray::int32 dim = 3;

  using Query = dray::Vec<dray::float32,dim>;
  using Elem = dray::MeshElem<float32, dim>;
  using Sol = dray::Vec<dray::float32, dim>;
  using RefBox = RefBox<dim>;

  struct FInBounds { DRAY_EXEC bool operator()(const Query &query, const Elem &elem, const RefBox &ref_box) {
    /// fprintf(stderr, "FInBounds callback\n");
    dray::AABB<> bounds;
    elem.get_sub_bounds(ref_box.begin(), &bounds.m_x);
    fprintf(stderr, "  aabb==[%.4f,%.4f,  %.4f,%.4f,  %.4f,%.4f]\n",
        bounds.m_x.min(), bounds.m_x.max(),
        bounds.m_y.min(), bounds.m_y.max(),
        bounds.m_z.min(), bounds.m_z.max() );
    return ( bounds.m_x.min() <= query[0] && query[0] < bounds.m_x.max()  &&
             bounds.m_y.min() <= query[1] && query[1] < bounds.m_y.max()  &&
             bounds.m_z.min() <= query[2] && query[2] < bounds.m_z.max() );
  } };

  struct FGetSolution { DRAY_EXEC bool operator()(const Query &query, const Elem &elem, const RefBox &ref_box, Sol &solution) {
    /// fprintf(stderr, "FGetSolution callback\n");
    solution = ref_box.template center<dray::float32>();   // Awesome initial guess. TODO also use ref_box to guide the iteration.
    return elem.eval_inverse(query, solution, true);
  } };

  RefBox ref_box;
  ref_box[0].include(0.0);  ref_box[0].include(1.0);
  ref_box[1].include(0.0);  ref_box[1].include(1.0);
  ref_box[2].include(0.0);  ref_box[2].include(1.0);
  Sol solution;

  Elem elem;
  constexpr dray::int32 num_dofs = dray::intPow(1+p_order, dim);
  dray::Vec<dray::float32,dim> val_list[num_dofs] =                 // Identity map.
  {
    {0.0, 0.0, 0.0},
    {0.0, 0.0, 0.5},
    {0.0, 0.0, 1.0},

    {0.0, 0.5, 0.0},
    {0.0, 0.5, 0.5},
    {0.0, 0.5, 1.0},

    {0.0, 1.0, 0.0},
    {0.0, 1.0, 0.5},
    {0.0, 1.0, 1.0},


    {0.5, 0.0, 0.0},
    {0.5, 0.0, 0.5},
    {0.5, 0.0, 1.0},

    {0.5, 0.5, 0.0},
    {0.5, 0.5, 0.5},
    {0.5, 0.5, 1.0},

    {0.5, 1.0, 0.0},
    {0.5, 1.0, 0.5},
    {0.5, 1.0, 1.0},


    {1.0, 0.0, 0.0},
    {1.0, 0.0, 0.5},
    {1.0, 0.0, 1.0},

    {1.0, 0.5, 0.0},
    {1.0, 0.5, 0.5},
    {1.0, 0.5, 1.0},

    {1.0, 1.0, 0.0},
    {1.0, 1.0, 0.5},
    {1.0, 1.0, 1.0},
  };
  dray::int32 ctrl_idx_list[num_dofs] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26};

  elem.construct(0, p_order, ctrl_idx_list, val_list);

  Query query = {0.7, 0.7, 0.7};

  /// {
  ///   dray::Vec<dray::float32, dim> result_val;
  ///   dray::Vec<dray::Vec<dray::float32, dim>, dim> result_deriv;
  ///   elem.eval({0.5, 0.5, 0.5}, result_val, result_deriv);
  ///   fprintf(stderr, "Elem eval: (%.4f, %.4f, %.4f)\n", result_val[0], result_val[1], result_val[2]);
  /// }

  auto ret_code = dray::SubdivisionSearch::subdivision_search<Query, Elem, RefBox, Sol, FInBounds, FGetSolution>(
      query, elem, &ref_box, &solution, 1);

  // Report results.
  fprintf(stderr, "Solution: (%f, %f, %f)\n", solution[0], solution[1], solution[2]);
}
