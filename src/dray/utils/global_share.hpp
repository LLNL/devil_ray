#ifndef DRAY_GLOBAL_SHARE_HPP
#define DRAY_GLOBAL_SHARE_HPP

#include <memory>

namespace dray
{

  // GlobalShare: Utility class/workaround to create global access to objects
  //   that cannot live directly at global scope, e.g. dray::Array.
  //
  // DO NOT store handles from this class in other global-lifetime objects,
  //   or else the same bad things will happen.
  //
  // Note: type T must be default-constructable.
  //
  template <class T>
  class GlobalShare
  {
    public:
      // Get a global handle; object will be destroyed after the last handle is destructed.
      //   As long as each handle is locally scoped inside some function, the global object
      //   will be destroyed before main() returns.
      std::shared_ptr<T> get_shared_ptr()
      {
        std::shared_ptr<T> ret;
        if (!m_glob_weak_ptr.use_count())
        {
          ret = std::make_shared<T>();
          m_glob_weak_ptr = ret;
        }
        else
          ret = m_glob_weak_ptr.lock();

        return ret;
      }

    private:
      // Only a weak pointer lives at global scope.
      std::weak_ptr<T> m_glob_weak_ptr;
  };

}

#endif//DRAY_GLOBAL_SHARE_HPP
