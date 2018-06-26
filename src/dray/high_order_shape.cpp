#include <dray/high_order_shape.hpp>
#include <dray/exports.hpp>
#include <dray/policies.hpp>
#include <dray/arrayvec.hpp>

#include <assert.h>

namespace dray
{

namespace detail
{
  // Explicit instantiations.
  class BernsteinShape<float32,3,2>;   // Quadratic volume
  class BernsteinShape<float32,1,2>;   // Quadratic curve
  class BernsteinShape<float32,3,0>;   // Point
  class BernsteinShape<float32,1,0>;   // Point
} // namespace detail


template <typename T, int32 C, int32 DOF, typename ShapeFunctor, int32 D>
ArrayVec<T,C>
FunctionCtrlPoints<T,C,DOF, ShapeFunctor,D>::eval(const ShapeFunctor &_shape_f, const ArrayVec<T,D> &ref_pts) const
{
  /// // Check that shape_f has the right dimensions (compile time).
  /// const ShapeDims<D,DOF> cmpl_test_shape_dims = ShapeFunctor::shape_dims;

  const int32 num_elts = m_ctrl_idx.size() / DOF;
  assert(ref_pts.size() == num_elts);

  const ShapeFunctor shape_f = _shape_f;  // local for lambda capture.
  const RefVec *ref_pts_ptr = ref_pts.get_device_ptr_const();
  const PhysVec *values_ptr = m_values.get_device_ptr_const();
  const int32 *ctrl_idx_ptr = m_ctrl_idx.get_device_ptr_const();

  Array<PhysVec> elt_vals_out;
  elt_vals_out.resize(num_elts);
  PhysVec *elt_vals_ptr = elt_vals_out.get_device_ptr();

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, num_elts), [=] DRAY_LAMBDA (int32 elt_idx)
  ///RAJA::forall<for_cpu_policy>(RAJA::RangeSegment(0, num_elts), [=] (int32 elt_idx)
  {
    // Compute element shape values.
    ShapeVec elt_shape;
    shape_f(ref_pts_ptr[elt_idx], elt_shape);

    // Grab and accumulate the control point values for this element,
    // weighted by the shape values.
    // (This part is sequential and not parallel because it is a "segmented reduction.")
    PhysVec elt_val = static_cast<T>(0.f);
    for (int32 dof_idx = 0; dof_idx < DOF; dof_idx++)
    {
      elt_val += elt_shape[dof_idx] * values_ptr[ctrl_idx_ptr[elt_idx*DOF + dof_idx]];
    }

    elt_vals_ptr[elt_idx] = elt_val;
  });

  return elt_vals_out;
}

// Explicit instantiations.

template class FunctionCtrlPoints<float32, 1,27, detail::DummyUniformShape<float32,3,27>, 3>;

} // namespace dray
