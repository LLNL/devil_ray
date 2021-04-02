#include <dray/queries/lineout_3d.hpp>

#include <dray/dispatcher.hpp>
#include <dray/Element/elem_utils.hpp>
#include <dray/GridFunction/mesh.hpp>
#include <dray/GridFunction/device_mesh.hpp>
#include <dray/GridFunction/mesh_utils.hpp>
#include <dray/utils/data_logger.hpp>

#include <dray/policies.hpp>
#include <dray/error_check.hpp>
#include <RAJA/RAJA.hpp>


namespace dray
{

namespace detail
{
#if 0
template<typename MeshElem>
Array<Vec<Float,3>>
lineout_locate_execute(Mesh<MeshElem> &mesh,
                       Array<Vec<Float,3>> points)
{
  DRAY_LOG_OPEN("lineout_locate");
  Array<Vec<Float,3>> ref_points;


  DRAY_LOG_CLOSE();
  return ref_points;
}
#endif

struct LineoutLocateFunctor
{
  Array<Vec<Float,3>> m_points;
  Array<Vec<Float,3>> m_ref_points;
  LineoutLocateFunctor(const Array<Vec<Float,3>> &points)
    : m_points(points)
  {
  }

  template<typename TopologyType>
  void operator()(TopologyType &topo)
  {
    //m_res = detail::reflect_execute(topo.mesh(), m_point, m_normal);
  }
};

}//namespace detail

Lineout3D::Lineout3D()
  : m_samples(100),
    m_empty_val(0)
{
}

int32
Lineout3D::samples() const
{
  return m_samples;
}

void
Lineout3D::samples(int32 samples)
{
  if(samples < 1)
  {
    DRAY_ERROR("Number of samples must be positive");
  }
  m_samples = samples;
}

void
Lineout3D::add_line(const Vec<Float,3> start, const Vec<Float,3> end)
{
  m_starts.push_back(start);
  m_ends.push_back(end);
}

void
Lineout3D::add_var(const std::string var)
{
  m_vars.push_back(var);
}

void
Lineout3D::empty_val(const Float val)
{
  m_empty_val = val;
}

Array<Vec<Float,3>>
Lineout3D::create_points()
{
  const int32 lines_size = m_starts.size();
  // number of samples will be at the beginning + end of
  // the line plus how ever many samples the user asks for
  // m_samples must be > 0
  const int32 samples = m_samples + 2;
  const int32 total_points = lines_size * samples;

  Array<Vec<Float,3>> starts;
  Array<Vec<Float,3>> ends;
  starts.resize(lines_size);
  ends.resize(lines_size);
  // pack the points
  {
    Vec<Float,3> *starts_ptr = starts.get_host_ptr();
    Vec<Float,3> *ends_ptr = ends.get_host_ptr();
    for(int32 i = 0; i < lines_size; ++i)
    {
      starts_ptr[i] = m_starts[i];
      ends_ptr[i] = m_ends[i];
    }
  }

  Array<Vec<Float,3>> points;
  points.resize(total_points);
  Vec<Float,3> *points_ptr = points.get_device_ptr();

  const Vec<Float,3> *starts_ptr = starts.get_device_ptr_const();
  const Vec<Float,3> *ends_ptr = ends.get_device_ptr_const();

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, total_points), [=] DRAY_LAMBDA (int32 i)
  {
    const int32 line_id = i / samples;
    const int32 sample_id = i % samples;
    const Vec<Float,3> start = starts_ptr[line_id];
    const Vec<Float,3> end = starts_ptr[line_id];
    Vec<Float,3> dir = start - end;
    const Float step = dir.magnitude() / Float(samples - 1);
    dir.normalize();
    Vec<Float,3> point = start + (step * Float(sample_id)) * dir;

    points_ptr[i] = point;
  });
  DRAY_ERROR_CHECK();

  return points;
}

void
Lineout3D::execute(Collection &collection)
{
  const int32 vars_size = m_vars.size();
  if(vars_size == 0)
  {
    DRAY_ERROR("Lineout3D: must specify at least 1 variables:");
  }
  const int32 lines_size = m_starts.size();
  if(lines_size == 0)
  {
    DRAY_ERROR("Lineout3D: must specify at least 1 line:");
  }

  for(int32 i = 0; i < vars_size; ++i)
  {
    if(!collection.has_field(m_vars[i]))
    {
      DRAY_ERROR("Lineout3D: no variable named '"<<m_vars[i]<<"'");
    }
  }

  Array<Vec<Float,3>> points = create_points();

  for(int32 i = 0; i < collection.local_size(); ++i)
  {
    DataSet data_set = collection.domain(i);
    Array<Location> locs = data_set.topology()->locate(points);
    //detail::LineoutLocateFunctor func(points);
    //dispatch(data_set.topology(), func);

    //// pass through all in the input fields
    //const int num_fields = data_set.number_of_fields();
    //for(int i = 0; i < num_fields; ++i)
    //{
    //  func.m_res.add_field(data_set.field_shared(i));
    //}
    //res.add_domain(func.m_res);
  }
}


}//namespace dray
