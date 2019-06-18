#ifndef DRAY_ELEMENT_HPP
#define DRAY_ELEMENT_HPP

#include <dray/el_trans.hpp>
#include <dray/bernstein_basis.hpp>
#include <dray/vec.hpp>
#include <dray/range.hpp>
#include <dray/exports.hpp>
#include <dray/types.hpp>

namespace dray
{

  // (2019-05-23) TODO right now this simply hides the ugliness of ElTransOp and family.
  //   Eventually this class should take the place of ElTransOp.
  template <typename T, unsigned int RefDim, unsigned int PhysDim>
  using ElementBase = ElTransOp<T, BernsteinBasis<T,RefDim>, ElTransIter<T,PhysDim>>;

  template <typename T, unsigned int PhysDim>
  class FaceElement;

  template <typename T, unsigned int RefDim, unsigned int PhysDim>
  class Element
  {
    public:
      using Base = ElementBase<T,RefDim,PhysDim>;
      static constexpr int32 phys_dim = PhysDim;    // Seen by NewtonSolve
      static constexpr int32 ref_dim = RefDim;      // Seen by NewtonSolve

      friend class FaceElement<T,PhysDim>;

      //
      // create() : factory method.
      DRAY_EXEC static Element create(int32 el_id, int32 poly_order, const int32 *ctrl_idx_ptr, const Vec<T,PhysDim> *val_ptr);
      
      //
      // construct() : constructor you must call explicitly.
      DRAY_EXEC void construct(int32 el_id, int32 poly_order, const int32 *ctrl_idx_ptr, const Vec<T,PhysDim> *val_ptr);

      //
      // get_bounds()
      DRAY_EXEC void get_bounds(Range *ranges) const;

      //TODO a set of evaluator functions, v, v_d, pv, jac. For now use ElTransOp::eval().

      DRAY_EXEC void eval(const Vec<T,RefDim> &ref, Vec<T,PhysDim> &result_val,
                          Vec<Vec<T,PhysDim>,RefDim> &result_deriv) const
      {
        m_base.eval(ref, result_val, result_deriv);
      }

      //
      // is_inside()
      DRAY_EXEC static bool is_inside(const Vec<T,RefDim> &ref_coords);

      //
      // get_el_id()
      DRAY_EXEC int32 get_el_id() const { return m_el_id; }

    protected:
      int32 m_el_id;
      mutable Base m_base;  // Composition rather than public/private inheritance, because const means different things.
  };


  template <typename T, unsigned int PhysDim>
  using FaceElementBase = ElTransOp<T, BernsteinBasis<T,2>, ElTransBdryIter<T,PhysDim>>;

  //
  // FaceElement
  //
  // Face of 3D hex.
  template <typename T, unsigned int PhysDim>
  class FaceElement
  {
    public:
      using Base = FaceElementBase<T,PhysDim>;

        // lowercase: 0_end. Uppercase: 1_end.
      enum class FaceID { x = 0, y = 1, z = 2, X = 3, Y = 4, Z = 5 };

      //
      // create() : factory method.
      DRAY_EXEC static FaceElement create(const Element<T,3,PhysDim> &host_elem, FaceID face_id);

      //
      // construct() : constructor you must call explicitly.
      DRAY_EXEC void construct(const Element<T,3,PhysDim> &host_elem, FaceID face_id);

      //
      // get_bounds()
      DRAY_EXEC void get_bounds(Range *ranges) const;

      //
      // set_face_coord() : Set the constant (reference) coordinate value of the face-normal axis.
      DRAY_EXEC void set_face_coordinate(Vec<T,3> &ref_coords)
      {
        switch (m_face_id)
        {
          case FaceID::x: case FaceID::X: ref_coords[0] = ((int32) m_face_id / 3); break;
          case FaceID::y: case FaceID::Y: ref_coords[1] = ((int32) m_face_id / 3); break;
          case FaceID::z: case FaceID::Z: ref_coords[2] = ((int32) m_face_id / 3); break;
        }
      }

      //
      // ref2fref() : Project by dropping the non-face coordinate.
      DRAY_EXEC void ref2fref(const Vec<T,3> &ref_coords, Vec<T,2> &fref_coords)
      {
        switch (m_face_id)
        {
          case FaceID::x: case FaceID::X: fref_coords = {ref_coords[1], ref_coords[2]}; break;
          case FaceID::y: case FaceID::Y: fref_coords = {ref_coords[0], ref_coords[2]}; break;
          case FaceID::z: case FaceID::Z: fref_coords = {ref_coords[0], ref_coords[1]}; break;
        }
      }

      //
      // fref2ref() : Embed by only setting the tangent coordinates.
      //              Might want to use with set_face_coord() other coordinate.
      DRAY_EXEC void fref2ref(const Vec<T,2> &fref_coords, Vec<T,3> &ref_coords)
      {
        switch (m_face_id)
        {
          case FaceID::x: case FaceID::X: ref_coords[1] = fref_coords[0];  ref_coords[2] = fref_coords[1]; break;
          case FaceID::y: case FaceID::Y: ref_coords[0] = fref_coords[0];  ref_coords[2] = fref_coords[1]; break;
          case FaceID::z: case FaceID::Z: ref_coords[0] = fref_coords[0];  ref_coords[1] = fref_coords[1]; break;
        }
      }


      //TODO a set of evaluator functions, v, v_d, pv, jac. For now use ElTransOp::eval().

      // For now, this evaluator accepts a 3-point, but ignores the off-plane coordinate.
      // The derivatives returned correspond to the two on-plane coordinates.
      DRAY_EXEC void eval(const Vec<T,3> &ref, Vec<T,PhysDim> &result_val,
                          Vec<T,PhysDim> &result_deriv_0,
                          Vec<T,PhysDim> &result_deriv_1) const
      {
        Vec<T,2> fref;
        ref2fref(ref, fref);

        Vec<Vec<T,PhysDim>, 2> result_deriv;
        eval(fref, result_val, result_deriv);  // see below.

        result_deriv_0 = result_deriv[0];
        result_deriv_1 = result_deriv[1];
      }

      // For this version the caller can use whichever coordinates were given, if it doesn't know.
      DRAY_EXEC void eval(const Vec<T,2> &fref, Vec<T,PhysDim> &result_val, Vec<Vec<T,PhysDim>,2> &result_deriv) const
      {
        m_base.eval(fref, result_val, result_deriv);
      }

      //
      // is_inside()
      DRAY_EXEC bool is_inside(const Vec<T,2> &fref_coords) const;

      DRAY_EXEC bool is_inside(const Vec<T,3> &ref_coords) const
      {
        Vec<T,2> fref;
        ref2fref(ref_coords, fref);
        return is_inside(fref);
      }


      //
      // get_el_id()
      DRAY_EXEC int32 get_el_id() const { return m_el_id; }

    protected:
      int32 m_el_id;
      FaceID m_face_id;
      mutable Base m_base;
  };

}



// Implementations
namespace dray
{

  //
  // create() : factory method.
  template <typename T, unsigned int RefDim, unsigned int PhysDim>
  DRAY_EXEC Element<T,RefDim,PhysDim> Element<T,RefDim,PhysDim>::create(int32 el_id, int32 poly_order,
      const int32 *ctrl_idx_ptr, const Vec<T,PhysDim> *val_ptr)
  {
    Element<T,RefDim,PhysDim> ret;
    ret.construct(el_id, poly_order, ctrl_idx_ptr, val_ptr);
    return ret;
  }

  //
  // construct() : constructor you must call explicitly.
  template <typename T, unsigned int RefDim, unsigned int PhysDim>
  DRAY_EXEC void Element<T,RefDim,PhysDim>::construct(
      int32 el_id, int32 poly_order,
      const int32 *ctrl_idx_ptr, const Vec<T,PhysDim> *val_ptr)
  {
    m_base.init_shape(poly_order);
    m_base.m_coeff_iter.init_iter(ctrl_idx_ptr, val_ptr, intPow(poly_order+1, RefDim), el_id);
    m_el_id = el_id;
  }

  //
  // get_bounds()
  template <typename T, unsigned int RefDim, unsigned int PhysDim>
  DRAY_EXEC void Element<T,RefDim,PhysDim>::get_bounds(Range *ranges) const
  {
    ElTransData<T, PhysDim>::get_elt_node_range(m_base.m_coeff_iter, m_base.get_el_dofs(), ranges);
  }


  //
  // is_inside()
  template <typename T, unsigned int RefDim, unsigned int PhysDim>
  DRAY_EXEC bool Element<T,RefDim,PhysDim>::is_inside(const Vec<T,RefDim> &ref_coords)
  {
    for (int32 d = 0; d < RefDim; d++)
      if (!(0.0 <= ref_coords[d] && ref_coords[d] < 1.0))
        return false;
    return true;
  }




  //
  // create() : factory method.
  template <typename T, unsigned int PhysDim>
  DRAY_EXEC FaceElement<T,PhysDim> FaceElement<T,PhysDim>::create(const Element<T,3,PhysDim> &host_elem, FaceID face_id)
  {
    FaceElement<T,PhysDim> ret;
    ret.construct(host_elem, face_id);
    return ret;
  }

  //
  // construct() : constructor you must call explicitly.
  template <typename T, unsigned int PhysDim>
  DRAY_EXEC void FaceElement<T,PhysDim>::construct(const Element<T,3,PhysDim> &host_elem, FaceID face_id)
  {
    m_el_id = host_elem.m_el_id;
    m_face_id = face_id;

    int32 poly_order = host_elem.m_base.p;

    m_base.init_shape(poly_order);
    m_base.m_coeff_iter.init_iter(host_elem.m_base.m_coeff_iter.m_el_dofs_ptr,
                                  host_elem.m_base.m_coeff_iter.m_val_ptr,
                                  poly_order+1, (int32) m_face_id);
  }

  //
  // get_bounds()
  template <typename T, unsigned int PhysDim>
  DRAY_EXEC void FaceElement<T,PhysDim>::get_bounds(Range *ranges) const
  {
    ElTransData<T, PhysDim>::get_elt_node_range(m_base.m_coeff_iter, m_base.get_el_dofs(), ranges);
  }

  //
  // is_inside()
  template <typename T, unsigned int PhysDim>
  DRAY_EXEC bool FaceElement<T,PhysDim>::is_inside(const Vec<T,2> &r) const
  {
    return (0.0 <= r[0] && r[0] < 1.0) && (0.0 <= r[1] && r[1] < 1.0);
  }

}

#endif//DRAY_ELEMENT_HPP

