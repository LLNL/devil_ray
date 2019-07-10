#ifndef DRAY_MESH_HPP
#define DRAY_MESH_HPP

#include <dray/GridFunction/grid_function_data.hpp>
#include <dray/Element/element.hpp>
#include <dray/newton_solver.hpp>
#include <dray/subdivision_search.hpp>
#include <dray/aabb.hpp>
#include <dray/linear_bvh_builder.hpp>
#include <dray/ref_point.hpp>
#include <dray/vec.hpp>
#include <dray/exports.hpp>

#include <dray/utils/appstats.hpp>

namespace dray
{

  /*
   * @class MeshElem
   * @brief Hexahedral element with (p+1)^3 dofs representing a transformation in the Bernstein basis.
   */
  template <typename T, int32 dim = 3>
  class MeshElem : public Element<T,dim,dim>
  {
    public:
    using Element<T,dim,dim>::construct;

    DRAY_EXEC static MeshElem
    create(int32 el_id,
           int32 poly_order,
           const int32 *ctrl_idx_ptr,
           const Vec<T,dim> *val_ptr);

    DRAY_EXEC FaceElement<T,3>
    get_face_element(typename FaceElement<T,3>::FaceID face_id) const;

    DRAY_EXEC FaceElement<T,3>
    get_face_element(int32 face_id) const
    {
      return get_face_element((typename FaceElement<T,3>::FaceID) face_id);
    }

    /* Forward evaluation: See Element::eval()   (which for now is ElTransOp::eval(). */

    //
    // eval_inverse() : Try to locate the point in reference space. Return false if not contained.
    //
    // use_init_guess determines whether guess_domain is used or replaced by AABB::ref_universe().
    DRAY_EXEC bool
    eval_inverse(const Vec<T,dim> &world_coords,
                 const AABB<dim> &guess_domain,
                 Vec<T,dim> &ref_coords,
                 bool use_init_guess = false) const;

    DRAY_EXEC bool
    eval_inverse(stats::IterativeProfile &iter_prof,
                 const Vec<T,dim> &world_coords,
                 const AABB<dim> &guess_domain,
                 Vec<T,dim> &ref_coords,
                 bool use_init_guess = false) const;

    DRAY_EXEC bool
    eval_inverse_local(const Vec<T,dim> &world_coords,
                       Vec<T,dim> &ref_coords) const;

    DRAY_EXEC bool
    eval_inverse_local(stats::IterativeProfile &iter_prof,
                       const Vec<T,dim> &world_coords,
                       Vec<T,dim> &ref_coords) const;
  };

  // Utilities for 3D.
  template <typename T>
  void face_normal_and_position(const FaceElement<T,3> &face,
                                const Vec<T,3> &ref_coords,
                                Vec<T,3> &world_position,
                                Vec<T,3> &world_normal);

  template <typename T>
  void face_normal_and_position(const FaceElement<T,3> &face,
                                const Vec<T,2> &fref_coords,
                                Vec<T,3> &world_position,
                                Vec<T,3> &world_normal);


  /*
   * @class MeshAccess
   * @brief Device-safe access to a collection of elements (just knows about the geometry, not fields).
   */
  template <typename T, int32 dim = 3>
  struct MeshAccess
  {
    const int32 *m_idx_ptr;
    const Vec<T,dim> *m_val_ptr;
    const int32  m_poly_order;

    //
    // get_elem()
    DRAY_EXEC MeshElem<T,dim> get_elem(int32 el_idx) const;

    // world2ref()
    DRAY_EXEC bool
    world2ref(int32 el_idx,
              const Vec<T,dim> &world_coords,
              const AABB<dim> &guess_domain,
              Vec<T,dim> &ref_coords,
              bool use_init_guess = false) const;

    DRAY_EXEC bool
    world2ref(stats::IterativeProfile &iter_prof,
              int32 el_idx,
              const Vec<T,dim> &world_coords,
              const AABB<dim> &guess_domain,
              Vec<T,dim> &ref_coords,
              bool use_init_guess = false) const;
  };


  /*
   * @class Mesh
   * @brief Host-side access to a collection of elements (just knows about the geometry, not fields).
   */
  template <typename T, int32 dim = 3>
  class Mesh
  {
    public:
      Mesh() = delete;  // For now, probably need later.
      Mesh(const GridFunctionData<T,dim> &dof_data, int32 poly_order);

      struct ExternalFaces
      {
        BVH m_bvh;
        Array<Vec<int32,2>> m_faces;
      };
      //
      // access_device_mesh() : Must call this BEFORE capture to RAJA lambda.
      MeshAccess<T,dim> access_device_mesh() const;

      //
      // access_host_mesh()
      MeshAccess<T,dim> access_host_mesh() const;

      //
      // get_poly_order()
      int32 get_poly_order() const { return m_poly_order; }

      //
      // get_num_elem()
      int32 get_num_elem() const { return m_dof_data.get_num_elem(); }

      const BVH get_bvh() const;

      AABB<3> get_bounds() const;

      //
      // get_dof_data()  // TODO should this be removed?
      GridFunctionData<T,dim> get_dof_data() { return m_dof_data; }

      //
      // get_ref_aabbs()
      const Array<AABB<3>> & get_ref_aabbs() const { return m_ref_aabbs; }


    //
    // locate()
    //
    template <class StatsType>
    void locate(Array<int32> &active_indices,
                Array<Vec<T,3>> &wpoints,
                Array<RefPoint<T,3>> &rpoints,
                StatsType &stats) const;

    void locate(Array<int32> &active_indices,
                Array<Vec<T,3>> &wpoints,
                Array<RefPoint<T,3>> &rpoints) const
    {
#ifdef DRAY_STATS
      std::shared_ptr<stats::AppStats> app_stats_ptr =
        stats::global_app_stats.get_shared_ptr();
#else
      stats::NullAppStats n, *app_stats_ptr = &n;
#endif
      locate(active_indices, wpoints, rpoints, *app_stats_ptr);
    }
    ExternalFaces m_external_faces;
      protected:
        GridFunctionData<T,dim> m_dof_data;
        int32 m_poly_order;
        BVH m_bvh;
        Array<AABB<3>> m_ref_aabbs;
    };

}

// Implementations (could go in a .tcc file and include that at the bottom of .hpp)

namespace dray
{

  // ---------------- //
  // MeshElem methods //
  // ---------------- //

  template <typename T, int32 dim>
  DRAY_EXEC MeshElem<T,dim>
  MeshElem<T,dim>::create(int32 el_id,
                          int32 poly_order,
                          const int32 *ctrl_idx_ptr,
                          const Vec<T,dim> *val_ptr)
  {
    MeshElem<T,dim> ret;
    ret.construct(el_id, poly_order, ctrl_idx_ptr, val_ptr);
    return ret;
  }

  template <typename T, int32 dim>
  DRAY_EXEC FaceElement<T,3>
  MeshElem<T,dim>::get_face_element(typename FaceElement<T,3>::FaceID face_id) const
  {
    return FaceElement<T,3>::create(*this, face_id);
  }

  //TODO accept bounds on the solution.
  template <typename T, int32 dim>
  DRAY_EXEC bool
  MeshElem<T,dim>::eval_inverse(const Vec<T,dim> &world_coords,
                                const AABB<dim> &guess_domain,
                                Vec<T,dim> &ref_coords,
                                bool use_init_guess) const
  {
    stats::IterativeProfile iter_prof;   iter_prof.construct();
    return eval_inverse(iter_prof, world_coords, guess_domain, ref_coords, use_init_guess);
  }


  template <typename T, int32 dim>
  DRAY_EXEC bool
  MeshElem<T,dim>::eval_inverse_local(const Vec<T,dim> &world_coords,
                                    Vec<T,dim> &ref_coords) const
  {
    stats::IterativeProfile iter_prof;   iter_prof.construct();
    return eval_inverse_local(iter_prof, world_coords, ref_coords);
  }


  template <typename T, int32 dim>
  DRAY_EXEC bool
  MeshElem<T,dim>::eval_inverse_local(stats::IterativeProfile &iter_prof,
                                      const Vec<T,dim> &world_coords,
                                      Vec<T,dim> &ref_coords) const
  {
    using IterativeMethod = IterativeMethod<T>;

    // Newton step to solve inverse of geometric transformation (assuming good initial guess).
    struct Stepper {
      DRAY_EXEC typename IterativeMethod::StepStatus operator()
        (Vec<T,dim> &x) const
      {
        Vec<T,dim> delta_y;
        Vec<Vec<T,dim>,dim> j_col;
        Matrix<T,dim,dim> jacobian;
        m_transf.eval(x, delta_y, j_col);
        delta_y = m_target - delta_y;

        for (int32 rdim = 0; rdim < dim; rdim++)
          jacobian.set_col(rdim, j_col[rdim]);

        bool inverse_valid;
        Vec<T,dim> delta_x;
        delta_x = matrix_mult_inv(jacobian, delta_y, inverse_valid);

        if (!inverse_valid)
          return IterativeMethod::Abort;

        x = x + delta_x;
        return IterativeMethod::Continue;
      }

      Element<T,dim,dim> m_transf;
      Vec<T,dim> m_target;

    } stepper{ *this, world_coords};
    //TODO somewhere else in the program, figure out how to set the precision
    //based on the gradient and the image resolution.
    const T tol_ref = 1e-5f;
    const int32 max_steps = 100;

    // Find solution.
    bool found = (IterativeMethod::solve(iter_prof,
                                   stepper,
                                   ref_coords,
                                   max_steps,
                                   tol_ref) == IterativeMethod::Converged
      && Element<T,dim,dim>::is_inside(ref_coords));
    return found;
  }


  template <typename T, int32 dim>
  DRAY_EXEC bool
  MeshElem<T,dim>::eval_inverse(stats::IterativeProfile &iter_prof,
                                const Vec<T,dim> &world_coords,
                                const AABB<dim> &guess_domain,
                                Vec<T,dim> &ref_coords,
                                bool use_init_guess) const
  {
    using StateT = stats::IterativeProfile;
    using QueryT = Vec<T,dim>;
    using ElemT = MeshElem<T,dim>;
    using RefBoxT = AABB<dim>;
    using SolT = Vec<T,dim>;

    const T tol_refbox = 1e-2f;
    constexpr int32 subdiv_budget = 0;

    RefBoxT domain = (use_init_guess ? guess_domain : AABB<dim>::ref_universe());

    // For subdivision search, test whether the sub-element possibly contains the query point.
    // Strict test because the bounding boxes are approximate.
    struct FInBounds { DRAY_EXEC bool operator()(StateT &state, const QueryT &query, const ElemT &elem, const RefBoxT &ref_box) {
      dray::AABB<> bounds;
      elem.get_sub_bounds(ref_box.m_ranges, bounds);
      bool in_bounds = true;
      for (int d = 0; d < dim; d++)
        in_bounds = in_bounds && bounds.m_ranges[d].min() <= query[d] && query[d] < bounds.m_ranges[d].max();
      return in_bounds;
    } };

    // Get solution when close enough: Iterate using Newton's method.
    struct FGetSolution { DRAY_EXEC bool operator()(StateT &state, const QueryT &query, const ElemT &elem, const RefBoxT &ref_box, SolT &solution) {
      solution = ref_box.center();   // Awesome initial guess. TODO also use ref_box to guide the iteration.
      stats::IterativeProfile &iterations = state;
      return elem.eval_inverse_local(iterations, query, solution);
    } };

    // Initiate subdivision search.
    uint32 ret_code;
    int32 num_solutions = SubdivisionSearch::subdivision_search
        <StateT, QueryT, ElemT, T, RefBoxT, SolT, FInBounds, FGetSolution, subdiv_budget>(
        ret_code, iter_prof, world_coords, *this, tol_refbox, &domain, &ref_coords, 1);

    return num_solutions > 0;
  }

  // Utilities for 3D.
  template <typename T>
  void face_normal_and_position(const FaceElement<T,3> face,
                                const Vec<T,3> ref_coords,
                                Vec<T,3> &world_position,
                                Vec<T,3> &world_normal)
  {
    Vec<T,3> deriv0, deriv1;
    face.eval(ref_coords, world_position, deriv0, deriv1);
    world_normal = cross(deriv0, deriv1);
    //TODO choose the direction that points outward. probably need the other derivative for that.
    // For now, assume that the Jacobian is always positive, i.e. right-handedness is preserved.
    if (face.get_face_id() == FaceElement<T,3>::x ||
        face.get_face_id() == FaceElement<T,3>::z ||
        face.get_face_id() == FaceElement<T,3>::Y)
      world_normal = -world_normal;
    world_normal.normalize();
  }

  template <typename T>
  void face_normal_and_position(const FaceElement<T,3> face,
                                const Vec<T,2> fref_coords,
                                Vec<T,3> &world_position,
                                Vec<T,3> &world_normal)
  {
    Vec<Vec<T,3>,2> derivs;
    face.eval(fref_coords, world_position, derivs);
    world_normal = cross(derivs[0], derivs[1]);
    //TODO choose the direction that points outward. probably need the other derivative for that.
    // For now, assume that the Jacobian is always positive, i.e. right-handedness is preserved.
    if (face.get_face_id() == FaceElement<T,3>::x ||
        face.get_face_id() == FaceElement<T,3>::z ||
        face.get_face_id() == FaceElement<T,3>::Y)
      world_normal = -world_normal;
    world_normal.normalize();
  }





  // ------------------ //
  // MeshAccess methods //
  // ------------------ //

  //
  // get_elem()
  template <typename T, int32 dim>
  DRAY_EXEC MeshElem<T,dim>
  MeshAccess<T,dim>::get_elem(int32 el_idx) const
  {
    // We are just going to assume that the elements in the data store
    // are in the same position as their id, el_id==el_idx.
    MeshElem<T,dim> ret;
    ret.construct(el_idx, m_poly_order, m_idx_ptr, m_val_ptr);
    return ret;
  }

  //
  // world2ref()
  template <typename T, int32 dim>
  DRAY_EXEC bool
  MeshAccess<T,dim>::world2ref(int32 el_idx,
                               const Vec<T,dim> &world_coords,
                               const AABB<dim> &guess_domain,
                               Vec<T,dim> &ref_coords,
                               bool use_init_guess) const
  {
    return get_elem(el_idx).eval_inverse(world_coords,
                                         guess_domain,
                                         ref_coords,
                                         use_init_guess);
  }
  template <typename T, int32 dim>
  DRAY_EXEC bool
  MeshAccess<T,dim>::world2ref(stats::IterativeProfile &iter_prof,
                               int32 el_idx,
                               const Vec<T,dim> &world_coords,
                               const AABB<dim> &guess_domain,
                               Vec<T,dim> &ref_coords,
                               bool use_init_guess) const
  {
    return get_elem(el_idx).eval_inverse(iter_prof,
                                         world_coords,
                                         guess_domain,
                                         ref_coords,
                                         use_init_guess);
  }


  // ---------------- //
  // Mesh methods     //
  // ---------------- //

  //
  // access_device_mesh()
  template <typename T, int32 dim>
  MeshAccess<T,dim> Mesh<T,dim>::access_device_mesh() const
  {
    return { m_dof_data.m_ctrl_idx.get_device_ptr_const(),
             m_dof_data.m_values.get_device_ptr_const(),
             m_poly_order };
  }

  //
  // access_host_mesh()
  template <typename T, int32 dim>
  MeshAccess<T,dim> Mesh<T,dim>::access_host_mesh() const
  {
    return { m_dof_data.m_ctrl_idx.get_host_ptr_const(),
             m_dof_data.m_values.get_host_ptr_const(),
             m_poly_order };
  }

} // namespace dray


#endif//DRAY_MESH_HPP
