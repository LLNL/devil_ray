#ifndef DRAY_FIELD_HPP
#define DRAY_FIELD_HPP

#include <dray/GridFunction/grid_function_data.hpp>
#include <dray/Element/element.hpp>
#include <dray/vec.hpp>
#include <dray/exports.hpp>

namespace dray
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
    DRAY_EXEC static FieldElem create(int32 el_id, int32 poly_order, const int32 *ctrl_idx_ptr, const Vec<T,PhysDim> *val_ptr, T *aux_mem_ptr); //TODO get rid of aux_mem_ptr

    /* Forward evaluation: See Element::eval()   (which for now is ElTransOp::eval(). */
  };


  /*
   * @class FieldAccess
   * @brief Device-safe access to a collection of elements (just knows about the geometry, not fields).
   */
  template <typename T, int32 RefDim = 3, int32 PhysDim = 1>
  struct FieldAccess
  {
    const int32 *m_idx_ptr;
    const Vec<T,PhysDim> *m_val_ptr;
    const int32  m_poly_order;

    //
    // get_elem()
    DRAY_EXEC FieldElem<T,RefDim,PhysDim> get_elem(int32 el_idx, T* aux_mem_ptr) const;  //TODO get rid of aux_mem_ptr
  };
  

  /*
   * @class Field
   * @brief Host-side access to a collection of elements (just knows about the geometry, not fields).
   */
  template <typename T, int32 RefDim = 3, int32 PhysDim = 1>
  class Field
  {
    public:
      Field() = delete;  // For now, probably need later.
      Field(const GridFunctionData<T,PhysDim> &dof_data, int32 poly_order) : m_dof_data(dof_data), m_poly_order(poly_order) {}
      
      //
      // access_device_field() : Must call this BEFORE capture to RAJA lambda.
      FieldAccess<T,RefDim,PhysDim> access_device_field() const;

      //
      // access_host_field()
      FieldAccess<T,RefDim,PhysDim> access_host_field() const;

      //
      // get_poly_order()
      int32 get_poly_order() { return m_poly_order; }

      //
      // get_num_elem()
      int32 get_num_elem() { return m_dof_data.get_num_elem(); }

      //
      // get_dof_data()  // TODO should this be removed?
      GridFunctionData<T,PhysDim> get_dof_data() { return m_dof_data; }

    protected:
      GridFunctionData<T,PhysDim> m_dof_data;
      int32 m_poly_order;
  };

}







// Implementations (could go in a .tcc file and include that at the bottom of .hpp)

namespace dray
{

  // ---------------- //
  // FieldElem methods //
  // ---------------- //

  template <typename T, int32 RefDim, int32 PhysDim>
  DRAY_EXEC FieldElem<T,RefDim,PhysDim> FieldElem<T,RefDim,PhysDim>::create(int32 el_id, int32 poly_order, const int32 *ctrl_idx_ptr, const Vec<T,PhysDim> *val_ptr, T *aux_mem_ptr)
  {
    FieldElem<T,RefDim,PhysDim> ret;
    ret.construct(el_id, poly_order, ctrl_idx_ptr, val_ptr, aux_mem_ptr);
    return ret;
  }


  // ------------------ //
  // FieldAccess methods //
  // ------------------ //

  //
  // get_elem()
  template <typename T, int32 RefDim, int32 PhysDim>
  DRAY_EXEC FieldElem<T,RefDim,PhysDim> FieldAccess<T,RefDim,PhysDim>::get_elem(int32 el_idx, T* aux_mem_ptr) const
  {
    // We are just going to assume that the elements in the data store
    // are in the same position as their id, el_id==el_idx.
    FieldElem<T,RefDim,PhysDim> ret;
    ret.construct(el_idx, m_poly_order, m_idx_ptr, m_val_ptr, aux_mem_ptr);
    return ret;
  }

  // ---------------- //
  // Field methods     //
  // ---------------- //

  //
  // access_device_field()
  template <typename T, int32 RefDim, int32 PhysDim>
  FieldAccess<T,RefDim,PhysDim> Field<T,RefDim,PhysDim>::access_device_field() const
  {
    return { m_dof_data.m_ctrl_idx.get_device_ptr_const(),
             m_dof_data.m_values.get_device_ptr_const(), m_poly_order };
  }

  //
  // access_host_field()
  template <typename T, int32 RefDim, int32 PhysDim>
  FieldAccess<T,RefDim,PhysDim> Field<T,RefDim,PhysDim>::access_host_field() const
  {
    return { m_dof_data.m_ctrl_idx.get_host_ptr_const(),
             m_dof_data.m_values.get_host_ptr_const(),
             m_poly_order };
  }

} // namespace dray


#endif//DRAY_FIELD_HPP
