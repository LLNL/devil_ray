#ifndef DRAY_BLUEPRINT_READER_HPP
#define DRAY_BLUEPRINT_READER_HPP

#include <conduit.hpp>
#include <dray/data_set.hpp>

namespace dray
{

class BlueprintReader
{
  public:
  static DataSet<MeshElem<3u, Quad, General>>
  load (const std::string &root_file, const int cycle);
  static DataSet<MeshElem<3u, Quad, General>> load (const std::string &root_file);

  static DataSet<MeshElem<3u, Quad, General>>
  blueprint_to_dray (const conduit::Node &n_dataset);
};

} // namespace dray

#endif // DRAY_MFEM_READER_HPP
