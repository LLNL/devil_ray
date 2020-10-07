#ifndef DRAY_REDISTRIBUTE_HPP
#define DRAY_REDISTRIBUTE_HPP

#include <dray/collection.hpp>

namespace dray
{

class Redistribute
{
protected:
public:
  Redistribute();
  Collection execute(Collection &collection);
};

};//namespace dray

#endif// header gaurd
