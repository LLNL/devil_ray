#ifndef DRAY_FIELD_HPP
#define DRAY_FIELD_HPP

#include <dray/GridFunction/grid_function_data.hpp>
#include <dray/Element/element.hpp>
#include <dray/vec.hpp>
#include <dray/exports.hpp>

namespace dray
{

  template <typename T, uint32 dim, uint32 ncomp, ElemType etype, Order P>
  using FieldElem = Element<T, dim, ncomp, etype, P>;


  template <class ElemT, uint32 ncomp>
  struct FieldOn_
  {
    using get_type = Element<typename ElemT::get_precision,
                             ElemT::get_dim(),
                             ncomp,
                             ElemT::get_etype(),
                             ElemT::get_P()>;
  };


  //
  // FieldOn<>
  //
  // Get element type that is the same element type but different number of components.
  // E.g., make a scalar element type over a given mesh element type:
  //    using MeshElemT = MeshElem<float32, 3u, Quad, General>;
  //    using FieldElemT = FieldOn<MeshElemT, 1u>;
  template <class ElemT, uint32 ncomp>
  using FieldOn = typename FieldOn_<ElemT, ncomp>::get_type;


  namespace oldelement
  {

    /*
     * @class FieldElem
     * @brief Hexahedral element with (p+1)^3 dofs representing a transformation in the Bernstein basis.
     */
    template <typename T, int32 RefDim = 3, int32 PhysDim = 1>
    class FieldElem : public Element<T,RefDim,PhysDim>
    {
      public:
      using Element<T,RefDim,PhysDim>::construct;
      DRAY_EXEC static FieldElem create(int32 el_id, int32 poly_order, const int32 *ctrl_idx_ptr, const Vec<T,PhysDim> *val_ptr);

      /* Forward evaluation: See Element::eval()   (which for now is ElTransOp::eval(). */
    };

  }//namespace oldelement


  /*
   * @class FieldAccess
   * @brief Device-safe access to a collection of elements (just knows about the geometry, not fields).
   */
  template <typename T, class ElemT>
  struct FieldAccess
  {
    static constexpr auto dim = ElemT::get_dim();
    static constexpr auto ncomp = ElemT::get_ncomp();
    static constexpr auto etype = ElemT::get_etype();

    const int32 *m_idx_ptr;
    const Vec<T,ncomp> *m_val_ptr;
    const int32  m_poly_order;

    //
    // get_elem()
    DRAY_EXEC ElemT get_elem(int32 el_idx) const;
  };


  /*
   * @class Field
   * @brief Host-side access to a collection of elements (just knows about the geometry, not fields).
   */
  template <typename T, class ElemT>
  class Field
  {
    public:
      Field() = delete;  // For now, probably need later.
      Field(const GridFunctionData<T, ElemT::get_ncomp()> &dof_data,
            int32 poly_order);

      //
      // access_device_field() : Must call this BEFORE capture to RAJA lambda.
      FieldAccess<T,ElemT> access_device_field() const;

      //
      // access_host_field()
      FieldAccess<T,ElemT> access_host_field() const;

      //
      // get_poly_order()
      int32 get_poly_order() { return m_poly_order; }

      //
      // get_num_elem()
      int32 get_num_elem() { return m_dof_data.get_num_elem(); }

      //
      // get_dof_data()  // TODO should this be removed?
      GridFunctionData<T, ElemT::get_ncomp()> get_dof_data() { return m_dof_data; }

      Range<> get_range() const;  //TODO aabb

    protected:
      GridFunctionData<T, ElemT::get_ncomp()> m_dof_data;
      int32 m_poly_order;
      Range<> m_range;  //TODO aabb
  };

}


// Implementations (could go in a .tcc file and include that at the bottom of .hpp)

namespace dray
{

  namespace oldelement
  {
    // ---------------- //
    // FieldElem methods //
    // ---------------- //

    template <typename T, int32 RefDim, int32 PhysDim>
    DRAY_EXEC FieldElem<T,RefDim,PhysDim> FieldElem<T,RefDim,PhysDim>::create(int32 el_id, int32 poly_order, const int32 *ctrl_idx_ptr, const Vec<T,PhysDim> *val_ptr)
    {
      FieldElem<T,RefDim,PhysDim> ret;
      ret.construct(el_id, poly_order, ctrl_idx_ptr, val_ptr);
      return ret;
    }
  }


  // ------------------ //
  // FieldAccess methods //
  // ------------------ //

  //
  // get_elem()
  template <typename T, class ElemT>
  DRAY_EXEC ElemT FieldAccess<T,ElemT>::get_elem(int32 el_idx) const
  {
    // We are just going to assume that the elements in the data store
    // are in the same position as their id, el_id==el_idx.
    ElemT ret;
    SharedDofPtr<dray::Vec<T,ncomp>> dof_ptr{ElemT::get_num_dofs(m_poly_order)*el_idx + m_idx_ptr, m_val_ptr};
    ret.construct(el_idx, dof_ptr, m_poly_order);
    return ret;
  }

  // ---------------- //
  // Field methods     //
  // ---------------- //

  //
  // access_device_field()
  template <typename T, class ElemT>
  FieldAccess<T,ElemT> Field<T,ElemT>::access_device_field() const
  {
    return { m_dof_data.m_ctrl_idx.get_device_ptr_const(),
             m_dof_data.m_values.get_device_ptr_const(), m_poly_order };
  }

  //
  // access_host_field()
  template <typename T, class ElemT>
  FieldAccess<T,ElemT> Field<T,ElemT>::access_host_field() const
  {
    return { m_dof_data.m_ctrl_idx.get_host_ptr_const(),
             m_dof_data.m_values.get_host_ptr_const(),
             m_poly_order };
  }

} // namespace dray


#endif//DRAY_FIELD_HPP
