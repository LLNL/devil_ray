#ifndef DRAY_FIELD_HPP
#define DRAY_FIELD_HPP

#include <dray/GridFunction/grid_function_data.hpp>
#include <dray/Element/element.hpp>
#include <dray/vec.hpp>
#include <dray/exports.hpp>

namespace dray
{

  template <uint32 dim, uint32 ncomp, ElemType etype, Order P>
  using FieldElem = Element<dim, ncomp, etype, P>;


  template <class ElemT, uint32 ncomp>
  struct FieldOn_
  {
    using get_type = Element<ElemT::get_dim(),
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

  /*
   * @class FieldAccess
   * @brief Device-safe access to a collection of elements (just knows about the geometry, not fields).
   */
  template <class ElemT>
  struct FieldAccess
  {
    static constexpr auto dim = ElemT::get_dim();
    static constexpr auto ncomp = ElemT::get_ncomp();
    static constexpr auto etype = ElemT::get_etype();

    const int32 *m_idx_ptr;
    const Vec<Float,ncomp> *m_val_ptr;
    const int32  m_poly_order;

    //
    // get_elem()
    DRAY_EXEC ElemT get_elem(int32 el_idx) const;
  };


  /*
   * @class Field
   * @brief Host-side access to a collection of elements (just knows about the geometry, not fields).
   */
  template <class ElemT>
  class Field
  {
    public:
      Field() = delete;  // For now, probably need later.
      Field(const GridFunctionData<ElemT::get_ncomp()> &dof_data,
            int32 poly_order);

      //
      // access_device_field() : Must call this BEFORE capture to RAJA lambda.
      FieldAccess<ElemT> access_device_field() const;

      //
      // access_host_field()
      FieldAccess<ElemT> access_host_field() const;

      //
      // get_poly_order()
      int32 get_poly_order() const { return m_poly_order; }

      //
      // get_num_elem()
      int32 get_num_elem() const { return m_dof_data.get_num_elem(); }

      //
      // get_dof_data()  // TODO should this be removed?
      GridFunctionData<ElemT::get_ncomp()> get_dof_data() { return m_dof_data; }

      Range<> get_range() const;  //TODO aabb

    protected:
      GridFunctionData<ElemT::get_ncomp()> m_dof_data;
      int32 m_poly_order;
      Range<> m_range;  //TODO aabb
  };

}


// Implementations (could go in a .tcc file and include that at the bottom of .hpp)

namespace dray
{

  // ------------------ //
  // FieldAccess methods //
  // ------------------ //

  //
  // get_elem()
  template <class ElemT>
  DRAY_EXEC ElemT FieldAccess<ElemT>::get_elem(int32 el_idx) const
  {
    // We are just going to assume that the elements in the data store
    // are in the same position as their id, el_id==el_idx.
    ElemT ret;
    SharedDofPtr<Vec<Float,ncomp>> dof_ptr{ElemT::get_num_dofs(m_poly_order)*el_idx + m_idx_ptr, m_val_ptr};
    ret.construct(el_idx, dof_ptr, m_poly_order);
    return ret;
  }

  // ---------------- //
  // Field methods     //
  // ---------------- //

  //
  // access_device_field()
  template <class ElemT>
  FieldAccess<ElemT> Field<ElemT>::access_device_field() const
  {
    return { m_dof_data.m_ctrl_idx.get_device_ptr_const(),
             m_dof_data.m_values.get_device_ptr_const(), m_poly_order };
  }

  //
  // access_host_field()
  template <class ElemT>
  FieldAccess<ElemT> Field<ElemT>::access_host_field() const
  {
    return { m_dof_data.m_ctrl_idx.get_host_ptr_const(),
             m_dof_data.m_values.get_host_ptr_const(),
             m_poly_order };
  }

} // namespace dray


#endif//DRAY_FIELD_HPP
