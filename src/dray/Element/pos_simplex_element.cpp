#include <dray/Element/pos_simplex_element.hpp>

namespace dray
{

namespace newelement
{


template <typename T, uint32 dim>
DRAY_EXEC bool TriRefSpace<T,dim>::is_inside_domain(const Vec<T,dim> &ref_coords)
{
  return false;  //TODO
}

template <typename T, uint32 dim>
DRAY_EXEC void TriRefSpace<T,dim>::clamp_to_domain(Vec<T,dim> &ref_coords)
{
  //TODO
}

template <typename T, uint32 dim>
DRAY_EXEC Vec<T,dim> TriRefSpace<T,dim>::project_to_domain(const Vec<T,dim> &r1, const Vec<T,dim> &r2)
{
  return {0.0}; //TODO
}


// Template instantiations.
template class TriRefSpace<float, 2u>;
template class TriRefSpace<float, 3u>;
template class TriRefSpace<double, 2u>;
template class TriRefSpace<double, 3u>;


}//namespace newelement

}//namespace dray
