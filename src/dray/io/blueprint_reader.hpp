#ifndef DRAY_BLUEPRINT_READER_HPP
#define DRAY_BLUEPRINT_READER_HPP

#include <dray/data_set.hpp>

namespace dray
{

class BlueprintReader
{
public:
  static DataSet<float32, MeshElem<float32, 3u, Quad, General>> load32(const std::string &root_file, const int cycle = 0);
  static DataSet<float64, MeshElem<float64, 3u, Quad, General>> load64(const std::string &root_file, const int cycle = 0);
};

}

#endif//DRAY_MFEM_READER_HPP
