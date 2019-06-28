#ifndef DRAY_SUBDIVISION_SEARCH_HPP
#define DRAY_SUBDIVISION_SEARCH_HPP

#include <dray/bernstein_basis.hpp>

namespace dray
{

  namespace detail
  {
    template <typename RefBox>
    DRAY_EXEC void split_ref_box(int32 depth, const RefBox &parent, RefBox &first_child, RefBox &second_child);

    template <typename X>
    DRAY_EXEC bool stack_push(X stack[], int32 &stack_sz, const int32 stack_cap, const X &x)
    {
      bool ret;
      if (ret = (stack_sz < stack_cap))
        stack[stack_sz++] = x;
      return ret;
    }

    template <typename X>
    DRAY_EXEC bool stack_pop(X stack[], int32 &stack_sz, X * &xp)
    {
      bool ret;
      if (ret = (stack_sz))
        *xp = stack[--stack_sz];
      else
        xp = nullptr;
      return ret;
    }
  }

  struct SubdivisionSearch
  {
    enum RetCode { MaxStack = 1u, MaxSubdiv = 2u };

    // Signatures
    //   DRAY_EXEC bool FInBounds::operator()(const Query &, const Elem &, const RefBox &);
    //   DRAY_EXEC bool FGetSolution::operator()(const Query &, const Elem &, const RefBox &, Sol &);

    template <typename Query, typename Elem, typename RefBox, typename Sol, typename FInBounds, typename FGetSolution, int32 stack_cap = 10>
    DRAY_EXEC static uint32 subdivision_search(const Query &query, const Elem &elem, RefBox *ref_box, Sol *solutions, const int32 list_cap = 1)
    {
      // Simulate a continuation-passing recursive version in which we would pass
      //   PointInCell(Element E, Stack<Element> stack, List<Solution> sols)
      // with the invariant that E may be nil only if stack is empty.
      //
      // In continuation-passing style, all state is re-packaged into the arguments for
      // the inner function call(s). When implemented as an iterative procedure, the function
      // arguments are modified in place.
      //
      // In general there could be multiple exit points, hence the while(true).
      //
      uint32 ret_code = 0;

      RefBox stack[stack_cap];
      int32 stack_sz = 0;
      int32 sol_sz = 0;

      int32 depth_stack[stack_cap];
      int32 depth = 0;

      int32 subdiv_budget = 20;
      const int32 target_depth = 6;

      while (true)
      {
        if (ref_box != nullptr)
        {
#ifdef DEBUG_CPU_ONLY
          fprintf(stderr, "depth==%d subdiv_budget==%d stack_sz==%d ref_box==[%.4f,%.4f,  %.4f,%.4f,  %.4f,%.4f]\n",
              depth, subdiv_budget, stack_sz,
              ref_box->m_ranges[0].min(), ref_box->m_ranges[0].max(),
              ref_box->m_ranges[1].min(), ref_box->m_ranges[1].max(),
              ref_box->m_ranges[2].min(), ref_box->m_ranges[2].max());
#endif
          if (!subdiv_budget || depth == target_depth) /* ref_box is 'leaf' */
          {
            /* process leaf. i.e. decide whether to accept it. */
            Sol new_solution;
            bool is_solution = true;
            is_solution = FGetSolution()(query, elem, *ref_box, new_solution);

            if (is_solution)
              detail::stack_push(solutions, sol_sz, list_cap, new_solution);
            if (sol_sz == list_cap)
              return ret_code;  // Early exit.

            if (stack_sz)
              depth = depth_stack[stack_sz-1];
            detail::stack_pop(stack, stack_sz, ref_box); // return the continuation.
          }// is leaf

          else  /* ref_box is not 'leaf'. */
          {
            depth++;
            subdiv_budget--;
            RefBox first, second;
            detail::split_ref_box(depth-1, *ref_box, first, second); /// fprintf(stderr, "done splitting.\n");
            bool in_first = FInBounds()(query, elem, first);         /// fprintf(stderr, "got first bounds.\n");
            bool in_second = FInBounds()(query, elem, second);       /// fprintf(stderr, "got second bounds.\n");

            bool stack_full = false;
            if (in_first && in_second)
            {
              stack_full = !detail::stack_push(stack, stack_sz, stack_cap, second);
              if (stack_full)
                ret_code |= (uint32) MaxStack;
              else
                depth_stack[stack_sz-1] = depth;
            }

            if (in_first)
              *ref_box = first;  // return the continuation.
            else if (in_second)
              *ref_box = second; // return the continuation.
            else
            {
              if (stack_sz)
                depth = depth_stack[stack_sz-1];
              detail::stack_pop(stack, stack_sz, ref_box);  // return the continuation.
            }
          }// non leaf
        }
        else /*ref_box is nullptr*/
        {
          // Return solutions.
          return ret_code;
        }

      }
    }// function subdivision_search()

  };

  namespace detail
  {
    template <typename RefBox>
    DRAY_EXEC void split_ref_box(int32 depth, const RefBox &parent, RefBox &first_child, RefBox &second_child)
    {
      const int32 split_dim = depth % 3;  //TODO something more clever.
      const float32 alpha = 0.5;

      first_child = parent;
      second_child = parent;
      parent.m_ranges[split_dim].split(alpha, first_child.m_ranges[split_dim], second_child.m_ranges[split_dim]);
    }
  }


}

#endif//DRAY_SUBDIVISION_SEARCH_HPP
