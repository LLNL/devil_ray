#include <array_internals_base.hpp>
#include <array_registry.hpp>

namespace rtracer
{

ArrayInternalsBase::ArrayInternalsBase()
{
  ArrayRegistry::add_array(this);
}

ArrayInternalsBase::~ArrayInternalsBase()
{
  ArrayRegistry::remove_array(this);
}

} // namespace rtracer
