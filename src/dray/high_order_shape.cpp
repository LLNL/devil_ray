#include <dray/high_order_shape.hpp>
#include <dray/exports.hpp>
#include <dray/policies.hpp>

namespace dray
{

namespace detail
{
  class BernsteinShape<float32,2,3>;
} // namespace detail


template <typename T, int32 C, int32 DOF>
template<typename ShapeFunctor, int32 D>
Array<Vec<T,C>>
FunctionCtrlPoints<T,C,DOF>::eval(const ShapeFunctor &_shape_f, const Array<Vec<T,D>> &ref_pts)
{
  /// // Check that shape_f has the right dimensions (compile time).
  /// const ShapeDims<D,DOF> cmpl_test_shape_dims = ShapeFunctor::shape_dims;

  const int32 num_elts = m_ctrl_idx.size();
  assert(ref_pts.size() == num_elts);

  const ShapeFunctor shape_f = _shape_f;  // local for lambda capture.
  const Vec<T,D> *ref_pts_ptr = ref_pts.get_device_ptr_const();
  const Vec<T,C> *values_ptr = m_values.get_device_ptr_const();
  const int32 *ctrl_idx_ptr = m_ctrl_idx.get_device_ptr_const();

  Array<Vec<T,C>> elt_vals_out;
  elt_vals_out.resize(num_elts);
  Vec<T,C> *elt_vals_ptr = elt_vals_out.get_device_ptr();

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, num_elts), [=] DRAY_LAMBDA (int32 elt_idx)
  ///RAJA::forall<for_cpu_policy>(RAJA::RangeSegment(0, num_elts), [=] (int32 elt_idx)
  {
    // Compute element shape values.
    Vec<T,DOF> elt_shape;
    shape_f(ref_pts_ptr[elt_idx], elt_shape);

    // Grab and accumulate the control point values for this element,
    // weighted by the shape values.
    // (This part is sequential and not parallel because it is a "segmented reduction.")
    Vec<T,C> elt_val = static_cast<T>(0.f);
    for (int32 dof_idx = 0; dof_idx < DOF; dof_idx++)
    {
      elt_val += shape_f[dof_idx] * values_ptr[ctrl_idx_ptr[elt_idx*DOF + dof_idx]];
    }
  });

  return elt_vals_out;
}

} // namespace dray
