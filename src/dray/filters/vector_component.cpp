#include <dray/filters/vector_component.hpp>

#include <dray/dispatcher.hpp>
#include <dray/Element/elem_utils.hpp>
#include <dray/GridFunction/mesh.hpp>
#include <dray/GridFunction/device_mesh.hpp>
#include <dray/GridFunction/mesh_utils.hpp>
#include <dray/utils/data_logger.hpp>

#include <dray/policies.hpp>
#include <dray/error_check.hpp>
#include <RAJA/RAJA.hpp>

#include <memory>


namespace dray
{

namespace detail
{

template<typename ElemType>
std::shared_ptr<FieldBase>
vector_comp_execute(Field<ElemType> &field,
                    const int32 component)
{
  DRAY_LOG_OPEN("vector_component");

  GridFunction<ElemType::get_ncomp()> input_gf = field.get_dof_data();
  GridFunction<1> output_gf;
  // the output will have the same params as the input, just a different
  // values type
  output_gf.m_ctrl_idx = input_gf.m_ctrl_idx;
  output_gf.m_el_dofs = input_gf.m_el_dofs;
  output_gf.m_size_el = input_gf.m_size_el;
  output_gf.m_size_ctrl = input_gf.m_size_ctrl;
  output_gf.m_values.resize(input_gf.m_values.size());

  Vec<Float,ElemType::get_ncomp()> *in_ptr = input_gf.m_values.get_device_ptr();
  Vec<Float,1> *out_ptr = output_gf.m_values.get_device_ptr();
  const int size = input_gf.m_values.size();

  RAJA::forall<for_policy>(RAJA::RangeSegment(0, size), [=] DRAY_LAMBDA (int32 i)
  {
    out_ptr[i][0] = in_ptr[i][component];
  });

  using OutElemT = Element<ElemType::get_dim(),
                           1,
                           ElemType::get_etype(),
                           ElemType::get_P()>;

  Field<OutElemT> foutput(output_gf, field.order(), "");
  std::shared_ptr<FieldBase> output = std::make_shared<Field<OutElemT>>(foutput);

  DRAY_LOG_CLOSE();
  return output;
}

struct VectorComponentFunctor
{
  int32 m_component;
  std::shared_ptr<FieldBase> m_output;
  VectorComponentFunctor(const int32 component)
    : m_component(component)

  {

  }

  template<typename FieldType>
  void operator()(FieldType &field)
  {
    m_output = detail::vector_comp_execute(field, m_component);
  }
};
}//namespace detail

VectorComponent::VectorComponent()
  : m_component(-1)
{
}

void
VectorComponent::component(const int32 comp)
{
  if(comp < 0 || comp > 2)
  {
    DRAY_ERROR("Vector component must be in range [0,2] given '"<<comp<<"'");
  }
  m_component = comp;
}

void
VectorComponent::output_name(const std::string name)
{
  m_output_name = name;
}

void
VectorComponent::field(const std::string name)
{
  m_field_name = name;
}

Collection
VectorComponent::execute(Collection &collection)
{
  if(m_component == -1)
  {
    DRAY_ERROR("Component never set");
  }

  if(m_field_name == "")
  {
    DRAY_ERROR("Must specify an field name");
  }

  if(!collection.has_field(m_field_name))
  {
    DRAY_ERROR("No field named '"<<m_field_name<<"'");
  }

  if(m_output_name == "")
  {
    DRAY_ERROR("Must specify an output  field name");
  }

  Collection res;
  for(int32 i = 0; i < collection.local_size(); ++i)
  {
    DataSet data_set = collection.domain(i);
    detail::VectorComponentFunctor func(m_component);
    dispatch_vector(data_set.field(m_field_name), func);
    func.m_output->name(m_output_name);
    data_set.add_field(func.m_output);

    // pass through
    res.add_domain(data_set);
  }
  return res;
}

}//namespace dray
