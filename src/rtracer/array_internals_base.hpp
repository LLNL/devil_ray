#ifndef RTRACER_ARRAY_INTERNALS_BASE_HPP
#define RTRACER_ARRAY_INTERNALS_BASE_HPP

namespace rtracer
{

class ArrayInternalsBase
{
public:
  ArrayInternalsBase();
  virtual ~ArrayInternalsBase();
  virtual void release_device_ptr() = 0;
};

} // namespace rtracer

#endif
