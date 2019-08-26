#ifndef DRAY_BLUEPRINT_READER_HPP
#define DRAY_BLUEPRINT_READER_HPP

#include <dray/data_set.hpp>

namespace dray
{

class BlueprintReader
{
public:
  static DataSet<float32> load32(const std::string &root_file, const int cycle = 0);
  static DataSet<float64> load64(const std::string &root_file, const int cycle = 0);
};

}

#endif//DRAY_MFEM_READER_HPP
