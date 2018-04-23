#ifndef RTRACER_ARRAY_INTERNALS
#define RTRACER_ARRAY_INTERNALS

#include<array_internals_base.hpp>

#include<umpire/Umpire.hpp>

#include<assert.h>

namespace rtracer
{

template<typename T>
class ArrayInternals : public ArrayInternalsBase
{
protected:
  T      *m_device;
  T      *m_host;
  bool    m_device_dirty;
  bool    m_host_dirty;
  size_t  m_size; 
  bool    m_cuda_enabled;
public:
  ArrayInternals()
    : ArrayInternalsBase(),
      m_device(nullptr),
      m_host(nullptr),
      m_device_dirty(true),
      m_host_dirty(true),
      m_size(0),
      m_cuda_enabled(false)
  {}
  
  void resize(const size_t size)
  {
    assert(size > 0);
    
    if(size == m_size) return;
 
    m_host_dirty = true;
    m_device_dirty = true;
    
    deallocate_host();
    deallocate_device();
  }

  T* get_device_ptr()
  {

    if(!m_cuda_enabled) 
    {
      return get_host_ptr();
    }

    if(m_device == nullptr)
    {
      allocate_device();
    }

    if(m_device_dirty && m_host != nullptr)
    {
      synch_to_device();
    }

    m_host_dirty = true;
    return m_device;
  }
  
  T* get_host_ptr()
  {
    if(m_host == nullptr)
    {
      allocate_host();
    }
   
    if(m_cuda_enabled)
    {
      if(m_host_dirty && m_host != nullptr)
      {
        synch_to_host();
      }
    }

    m_device_dirty = true;

    return m_host;
  }
  
  virtual ~ArrayInternals() override
  {
   
  }
  
  
  virtual void release_device_ptr() override
  {
    auto& rm = umpire::ResourceManager::getInstance();

  }

protected:

    void deallocate_host()
    {
      if(m_host != nullptr)
      {
        auto& rm = umpire::ResourceManager::getInstance();
        rm.deallocate(m_host);
      }
    }
    
    void allocate_host()
    {
      if(m_host == nullptr)
      {
        auto& rm = umpire::ResourceManager::getInstance();
        umpire::Allocator host_allocator = rm.getAllocator("HOST");
        T* m_host = static_cast<T*>(host_allocator.allocate(m_size*sizeof(T)));
      }
    }
    
    void deallocate_device()
    {
      if(m_cuda_enabled)
      {
        if(m_device != nullptr)
        {
          auto& rm = umpire::ResourceManager::getInstance();
          rm.deallocate(m_device);
        }
      }
    }

    void allocate_device()
    {
      if(m_cuda_enabled)
      {
        if(m_device == nullptr)
        {
          auto& rm = umpire::ResourceManager::getInstance();
          umpire::Allocator device_allocator = rm.getAllocator("DEVICE");
          T* m_host = static_cast<T*>(device_allocator.allocate(m_size*sizeof(T)));
        }
      }
    }
    
    // synchs assumes that both arrays are allocated
    void synch_to_host()
    {
      auto& rm = umpire::ResourceManager::getInstance();
      rm.copy(m_device, m_host);
    }

    void synch_to_device()
    {
      auto& rm = umpire::ResourceManager::getInstance();
      rm.copy(m_host, m_device);
    }

};

} // namespace rtracer

#endif
