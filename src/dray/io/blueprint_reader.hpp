#ifndef DRAY_BLUEPRINT_READER_HPP
#define DRAY_BLUEPRINT_READER_HPP

#include <dray/data_set.hpp>
#include <conduit.hpp>

namespace dray
{

class BlueprintReader
{
public:
  static DataSet<float32> load32(const std::string &root_file, const int cycle = 0);
  static DataSet<float64> load64(const std::string &root_file, const int cycle = 0);

  static DataSet<float32> blueprint_to_dray32(const conduit::Node &n_dataset);
  static DataSet<float64> blueprint_to_dray64(const conduit::Node &n_dataset);
};

}

#endif//DRAY_MFEM_READER_HPP
