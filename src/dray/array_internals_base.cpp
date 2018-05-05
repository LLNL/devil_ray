#include <dray/array_internals_base.hpp>
#include <dray/array_registry.hpp>

namespace dray
{

ArrayInternalsBase::ArrayInternalsBase()
{
  ArrayRegistry::add_array(this);
}

ArrayInternalsBase::~ArrayInternalsBase()
{
  ArrayRegistry::remove_array(this);
}

} // namespace dray
