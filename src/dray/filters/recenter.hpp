#ifndef DRAY_RECENTER_HPP
#define DRAY_RECENTER_HPP

#include <dray/collection.hpp>

namespace dray
{

class Recenter
{
  std::string m_field_name;
public:
  Recenter();
  Collection execute(Collection &collection);
  void field(const std::string field_name);
};

};//namespace dray

#endif//DRAY_RECENTER
