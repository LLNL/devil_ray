#include <dray/filters/path_lengths.hpp>
#include <dray/uniform_topology.hpp>
#include <dray/error.hpp>
#include <dray/policies.hpp>
#include <dray/utils/point_writer.hpp>
#include <dray/utils/png_encoder.hpp>
#include <dray/GridFunction/low_order_field.hpp>

namespace dray
{

namespace detail
{

static Array<Vec<Float,3>> cell_centers(UniformTopology &topo)
{

  const Vec<int32,3> cell_dims = topo.cell_dims();
  const Vec<Float,3> origin = topo.origin();
  const Vec<Float,3> spacing = topo.spacing();

  const int32 num_cells = cell_dims[0] * cell_dims[1] * cell_dims[2];

  Array<Vec<Float,3>> locations;
  locations.resize(num_cells);
  Vec<Float,3> *loc_ptr = locations.get_device_ptr();

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, num_cells), [=] DRAY_LAMBDA (int32 index)
  {
    Vec<int32,3> cell_id;
    cell_id[0] = index % cell_dims[0];
    cell_id[1] = (index / cell_dims[0]) % cell_dims[1];
    cell_id[2] = index / (cell_dims[0] * cell_dims[1]);

    Vec<Float,3> loc;
    for(int32 i = 0; i < 3; ++i)
    {
      loc[i] = origin[i] + Float(cell_id[i]) * spacing[i] + spacing[i] * 0.5f;
    }

    loc_ptr[index] = loc;
  });

  return locations;
}

struct TraversalState
{
  Vec<Float,3> m_delta_max;
  Vec<Float,3> m_delta;
  Vec<int32,3> m_voxel;
  Vec<Float,3> m_dir;

  // distance to voxel exit from initial point
  DRAY_EXEC
  Float exit() const
  {
    return min(m_delta_max[0], min(m_delta_max[1], m_delta_max[2]));
  }

  // advances to the next voxel along the ray
  DRAY_EXEC void advance()
  {
    int32 advance_dir = 0;
    for(int32 i = 1; i < 3; ++i)
    {
      if(m_delta_max[i] < m_delta_max[advance_dir])
      {
        advance_dir = i;
      }
    }
    m_delta_max[advance_dir] += m_delta[advance_dir];
    m_voxel[advance_dir] += m_dir[advance_dir] < 0.f ? -1 : 1;

    //std::cout<<"Voxel "<<voxel<<"\n";
    //assert(m_voxel[0] >= 0);
    //assert(m_voxel[1] >= 0);
    //assert(m_voxel[2] >= 0);

    //assert(m_voxel[0] < m_dims[0]);
    //assert(m_voxel[1] < m_dims[1]);
    //assert(m_voxel[2] < m_dims[2]);
  }

};

struct DDATraversal
{
  const Vec<int32,3> m_dims;
  const Vec<Float,3> m_origin;
  const Vec<Float,3> m_spacing;

  DDATraversal(UniformTopology &topo)
    : m_dims(topo.cell_dims()),
      m_origin(topo.origin()),
      m_spacing(topo.spacing())
  {

  }

  DRAY_EXEC
  bool is_inside(const Vec<int32, 3>& index) const
  {
    bool inside = true;
    const int32 minIndex = min(index[0], min(index[1], index[2]));
    if(minIndex < 0) inside = false;
    if(index[0] >= m_dims[0]) inside = false;
    if(index[1] >= m_dims[1]) inside = false;
    if(index[2] >= m_dims[2]) inside = false;
    return inside;
  }

  DRAY_EXEC
  int32 voxel_index(const Vec<int32, 3> &voxel) const
  {
    return voxel[0] + voxel[1] * m_dims[0] + voxel[2] * m_dims[0] * m_dims[1];
  }

  DRAY_EXEC Float
  init_traversal(const Vec<Float,3> &point,
                 const Vec<Float,3> &dir,
                 TraversalState &state) const
  {
    //assert(is_inside(point));
    Vec<Float, 3> temp = point;
    temp = temp - m_origin;
    state.m_voxel[0] = temp[0] / m_spacing[0];
    state.m_voxel[1] = temp[1] / m_spacing[1];
    state.m_voxel[2] = temp[2] / m_spacing[2];
    state.m_dir = dir;

    Vec<Float,3> step;
    step[0] = (dir[0] >= 0.f) ? 1.f : -1.f;
    step[1] = (dir[1] >= 0.f) ? 1.f : -1.f;
    step[2] = (dir[2] >= 0.f) ? 1.f : -1.f;

    Vec<Float,3> next_boundary;
    next_boundary[0] = (Float(state.m_voxel[0]) + step[0]) * m_spacing[0];
    next_boundary[1] = (Float(state.m_voxel[1]) + step[1]) * m_spacing[1];
    next_boundary[2] = (Float(state.m_voxel[2]) + step[2]) * m_spacing[2];

    // correct next boundary for negative directions
    if(step[0] == -1.f) next_boundary[0] += m_spacing[0];
    if(step[1] == -1.f) next_boundary[1] += m_spacing[1];
    if(step[2] == -1.f) next_boundary[2] += m_spacing[2];

    // distance to next voxel boundary
    state.m_delta_max[0] = (dir[0] != 0.f) ?
      (next_boundary[0] - (point[0] - m_origin[0])) / dir[0] : infinity<Float>();

    state.m_delta_max[1] = (dir[1] != 0.f) ?
      (next_boundary[1] - (point[1] - m_origin[1])) / dir[1] : infinity<Float>();

    state.m_delta_max[2] = (dir[2] != 0.f) ?
      (next_boundary[2] - (point[2] - m_origin[2])) / dir[2] : infinity<Float>();

    // distance along ray to traverse x,y, and z of a voxel
    state.m_delta[0] = (dir[0] != 0) ? m_spacing[0] / dir[0] * step[0] : infinity<Float>();
    state.m_delta[1] = (dir[1] != 0) ? m_spacing[1] / dir[1] * step[1] : infinity<Float>();
    state.m_delta[2] = (dir[2] != 0) ? m_spacing[2] / dir[2] * step[2] : infinity<Float>();

    Vec<Float,3> exit_boundary;
    exit_boundary[0] = step[0] < 0.f ? 0.f : Float(m_dims[0]) * m_spacing[0];
    exit_boundary[1] = step[1] < 0.f ? 0.f : Float(m_dims[1]) * m_spacing[1];
    exit_boundary[2] = step[2] < 0.f ? 0.f : Float(m_dims[2]) * m_spacing[2];

    if(step[0] == -1.f) exit_boundary[0] += m_spacing[0];
    if(step[1] == -1.f) exit_boundary[1] += m_spacing[1];
    if(step[2] == -1.f) exit_boundary[2] += m_spacing[2];

    Vec<Float,3> exit_dist;
    // distance to grid exit
    exit_dist[0] = (dir[0] != 0.f) ?
      (exit_boundary[0] - (point[0] - m_origin[0])) / dir[0] : infinity<Float>();

    exit_dist[1] = (dir[1] != 0.f) ?
      (exit_boundary[1] - (point[1] - m_origin[1])) / dir[1] : infinity<Float>();

    exit_dist[2] = (dir[2] != 0.f) ?
      (exit_boundary[2] - (point[2] - m_origin[2])) / dir[2] : infinity<Float>();

    //std::cout<<"Init voxel "<<voxel<<"\n";

    return min(exit_dist[0], min(exit_dist[1], exit_dist[2]));
  }



};


} // namespace detail

PathLengths::PathLengths()
 :  m_x_res(512),
    m_y_res(512),
    m_width(20.f),
    m_height(20.f),
    m_point({{0.f, 0.f, 15.f}}),
    m_normal({{0.f, 0.f, 1.f}}),
    m_x_dir({{1.f, 0.f, 0.f}})
{
}

Array<Vec<Float,3>>
PathLengths::generate_pixels()
{
  const Float pixel_width =  m_width / Float(m_x_res);
  const Float pixel_height = m_height / Float(m_y_res);

  // better be orthogonal
  assert(dot(m_normal, m_x_dir) == 0.f);

  Vec<Float,3> y_dir = cross(m_normal, m_x_dir);
  y_dir.normalize();
  // avoid lambda capturing 'this'
  Vec<Float,3> x_dir = m_x_dir;
  x_dir.normalize();
  const int32 width = m_x_res;

  Vec<Float,3> start;
  // we need the bottom left of the quad which is centered
  // at 'm_point'
  start = m_point - x_dir * m_width * 0.5f - y_dir * m_height * 0.5f;
  start += (pixel_width / 2.f) * x_dir;
  start += (pixel_height / 2.f) * y_dir;

  const int32 num_pixels = m_x_res * m_y_res;

  Array<Vec<Float,3>> pixels;
  pixels.resize(num_pixels);
  Vec<Float,3> *pixel_ptr = pixels.get_device_ptr();

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, num_pixels), [=] DRAY_LAMBDA (int32 index)
  {
    const int32 x = index % width;
    const int32 y = index / width;

    Vec<Float,3> loc;
    loc = start + (x * pixel_width * x_dir) + (y * pixel_height * y_dir);
    pixel_ptr[index] = loc;
  });

  return pixels;
}

Array<Float>
go(Array<Vec<Float,3>> &pixels,
   Array<Vec<Float,3>> &samples,
   UniformTopology &topo,
   LowOrderField *absorption,
   LowOrderField *emission)
{
  // input
  const detail::DDATraversal dda(topo);
  const Vec<Float,3> *pixel_ptr = pixels.get_device_ptr_const();
  const Vec<Float,3> *sample_ptr = samples.get_device_ptr_const();
  const int32 size_samples = samples.size();
  const int32 size = pixels.size();
  const Float *absorption_ptr = absorption->values().get_device_ptr_const();
  const Float *emission_ptr = emission->values().get_device_ptr_const();

  // output
  Array<Float> path_lengths;
  path_lengths.resize(size);
  Float *length_ptr = path_lengths.get_device_ptr();

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, size), [=] DRAY_LAMBDA (int32 index)
  {
    Vec<Float,3> pixel = pixel_ptr[index];
    Float sum = 0;
    for(int sample = 0; sample < size_samples; ++sample)
    {
      Vec<Float,3> loc = sample_ptr[sample];
      Vec<Float,3> dir = pixel - loc;
      dir.normalize();
      detail::TraversalState state;
      dda.init_traversal(loc, dir, state);

      Float distance = 0.f;
      Float res = 0.f;

      while(dda.is_inside(state.m_voxel))
      {
        const Float voxel_exit = state.exit();
        const Float length = voxel_exit - distance;

        const int32 cell_id = dda.voxel_index(state.m_voxel);
        const Float absorb = exp(-absorption_ptr[cell_id] * length);
        const Float emis = emission_ptr[cell_id];
        res = res * absorb + emis * (1.f - absorb);
        // this will get more complicated with MPI and messed up
        // metis domain decompositions

        distance = voxel_exit;
        state.advance();
        //if(index == 40)
        //{
        //  std::cout<<state.m_voxel<<" length "<<length<<" res "<<res<<" abs "<<absorb<<" emis "<<emis<<"\n";
        //  std::cout<<absorption_ptr[cell_id]<<" "<<emission_ptr[cell_id]<<"\n";
        //}

      }
      sum += res;
    }
    length_ptr[index] = sum;
  });
  return path_lengths;
}

void PathLengths::execute(DataSet &data_set)
{
  if(m_absorption_field == "")
  {
    DRAY_ERROR("Absorption field not set");
  }

  if(m_emission_field == "")
  {
    DRAY_ERROR("Emission field not set");
  }

  if(!data_set.has_field(m_absorption_field))
  {
    DRAY_ERROR("No absorption field '"<<m_absorption_field<<"' found");
  }

  if(!data_set.has_field(m_emission_field))
  {
    DRAY_ERROR("No emission field '"<<m_emission_field<<"' found");
  }

  Array<Vec<Float,3>> pixels = generate_pixels();
  write_points(pixels);

  TopologyBase *topo = data_set.topology();
  if(dynamic_cast<UniformTopology*>(topo) != nullptr)
  {
    std::cout<<"Boom\n";

    UniformTopology *uni_topo = dynamic_cast<UniformTopology*>(topo);
    LowOrderField *absorption = dynamic_cast<LowOrderField*>(data_set.field(m_absorption_field));
    LowOrderField *emission = dynamic_cast<LowOrderField*>(data_set.field(m_emission_field));

    if(absorption->assoc() != LowOrderField::Assoc::Element)
    {
      DRAY_ERROR("Absorption field must be associated with elements");
    }

    if(emission->assoc() != LowOrderField::Assoc::Element)
    {
      DRAY_ERROR("Emission field must be associated with elements");
    }
    Array<Vec<Float,3>> samples = detail::cell_centers(*uni_topo);
    Array<Float> plengths = go(pixels, samples, *uni_topo, absorption, emission);
    write_image(plengths);
  }
}

void PathLengths::write_image(Array<Float> values)
{
  const Float *values_ptr = values.get_device_ptr_const();
  const int32 size = values.size();

  RAJA::ReduceMin<reduce_policy, Float> xmin (infinity<Float>());
  RAJA::ReduceMax<reduce_policy, Float> xmax (neg_infinity<Float>());

  RAJA::forall<for_policy> (RAJA::RangeSegment (0, size), [=] DRAY_LAMBDA (int32 ii)
  {
    const Float value = values_ptr[ii];
    xmin.min (value);
    xmax.max (value);
  });

  float32 minv = xmin.get ();
  float32 maxv = xmax.get ();
  const float32 len = maxv - minv;
  const int32 image_size = m_x_res * m_y_res;

  Array<float32> dbuffer;
  dbuffer.resize (image_size * 4);

  float32 *d_ptr = dbuffer.get_host_ptr ();

  RAJA::forall<for_policy> (RAJA::RangeSegment (0, image_size), [=] DRAY_LAMBDA (int32 i)
  {
    const float32 depth = values_ptr[i];
    float32 value = 0.f;

    if (depth != infinity32 ())
    {
      value = (depth - minv) / len;
    }
    //std::cout<<value<<" ";
    const int32 offset = i * 4;
    d_ptr[offset + 0] = value;
    d_ptr[offset + 1] = value;
    d_ptr[offset + 2] = value;
    d_ptr[offset + 3] = 1.f;
  });

  PNGEncoder png_encoder;

  png_encoder.encode (d_ptr, m_x_res, m_y_res);

  png_encoder.save ("path_lengths.png");
}

void PathLengths::absorption_field(const std::string field_name)
{
  m_absorption_field = field_name;
}

void PathLengths::emission_field(const std::string field_name)
{
  m_emission_field = field_name;
}

void PathLengths::resolution(const int32 x, const int32 y)
{
  m_x_res = x;
  m_y_res = y;
}

void PathLengths::size(const float32 width, const float32 height)
{
  m_width = width;
  m_height = height;
}

void PathLengths::point(Vec<float32,3> p)
{
  m_point[0] = p[0];
  m_point[1] = p[1];
  m_point[2] = p[2];
}

};//namespace dray

