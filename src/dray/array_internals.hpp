#ifndef DRAY_ARRAY_INTERNALS
#define DRAY_ARRAY_INTERNALS

#include <dray/array_internals_base.hpp>
#include <dray/exports.hpp>

#include <umpire/Umpire.hpp>

#include <assert.h>
#include <iostream>

namespace dray
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
      m_size(0)
  { 
#ifdef CUDA_ENABLED
    m_cuda_enabled = true;
#else
    m_cuda_enabled = false;
#endif
  }
  
  size_t size()
  {
    return m_size;
  }
  void resize(const size_t size)
  {

    assert(size > 0);
    
    if(size == m_size) return;
 
    m_host_dirty = true;
    m_device_dirty = true;
    
    deallocate_host();
    deallocate_device();
    m_size = size;
  }

  T* get_device_ptr()
  {
    assert(m_size > 0);

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

    // indicate that the device has the most recent data
    m_host_dirty = true;
    m_device_dirty = false;
    return m_device;
  }
  
  const T* get_device_ptr_const()
  {
    assert(m_size > 0);
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

    m_device_dirty = false;
    return m_device;
  }
  
  T* get_host_ptr()
  {
    assert(m_size > 0);
    if(m_host == nullptr)
    {
      allocate_host();
    }
     
    if(m_cuda_enabled)
    {
      if(m_host_dirty && m_device != nullptr)
      {
        synch_to_host();
      }
    }

    // indicate that the host has the most recent data
    m_device_dirty = true;
    m_host_dirty = false;

    return m_host;
  }
  
  T* get_host_ptr_const()
  {
    assert(m_size > 0);
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

    m_host_dirty = false;

    return m_host;
  }
  
  virtual ~ArrayInternals() override
  {
   
  }
  
  //
  // Allow the release of device memory and save the 
  // existing data on the host if applicable
  //
  virtual void release_device_ptr() override
  {
    assert(m_size > 0);
    if(m_cuda_enabled)
    {
      if(m_device != nullptr)
      {

        if(m_host == nullptr)
        {
          allocate_host();
        }

        if(m_host_dirty)
        {
          synch_to_host();
        }
      }
    }

    deallocate_device();
    m_device_dirty = true;
    
  }

  virtual size_t device_alloc_size() override
  {
    if(m_device == nullptr) return 0;
    else return static_cast<size_t>(sizeof(T)) * m_size;
  } 

  virtual size_t host_alloc_size() override
  {
    if(m_host == nullptr) return 0;
    else return static_cast<size_t>(sizeof(T)) * m_size;
  } 
protected:

    void deallocate_host()
    {
      if(m_host != nullptr)
      {
        auto& rm = umpire::ResourceManager::getInstance();
        umpire::Allocator host_allocator = rm.getAllocator("HOST");
        host_allocator.deallocate(m_host);
        m_host = nullptr;
        m_host_dirty = true;
      }
    }
    
    void allocate_host()
    {
      assert(m_size > 0);
      if(m_host == nullptr)
      {
        auto& rm = umpire::ResourceManager::getInstance();
        umpire::Allocator host_allocator = rm.getAllocator("HOST");
        m_host = static_cast<T*>(host_allocator.allocate(m_size*sizeof(T)));
      }
    }
    
    void deallocate_device()
    {
      if(m_cuda_enabled)
      {
        if(m_device != nullptr)
        {
          auto& rm = umpire::ResourceManager::getInstance();
          umpire::Allocator device_allocator = rm.getAllocator("DEVICE");
          device_allocator.deallocate(m_device);
          m_device = nullptr;
          m_device_dirty = true;
        }
      }
    }

    void allocate_device()
    {
      assert(m_size > 0);
      if(m_cuda_enabled)
      {
        if(m_device == nullptr)
        {
          auto& rm = umpire::ResourceManager::getInstance();
          umpire::Allocator device_allocator = rm.getAllocator("DEVICE");
          m_device = static_cast<T*>(device_allocator.allocate(m_size*sizeof(T)));
        }
      }
    }
    
    // synchs assumes that both arrays are allocated
    void synch_to_host()
    {
      auto& rm = umpire::ResourceManager::getInstance();
      rm.copy(m_host, m_device);
    }

    void synch_to_device()
    {
      auto& rm = umpire::ResourceManager::getInstance();
      rm.copy(m_device, m_host);
    }

};

} // namespace dray

#endif
