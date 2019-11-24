#ifndef DRAY_MFEM_READER_HPP
#define DRAY_MFEM_READER_HPP

#include <dray/data_set.hpp>

namespace dray
{

class MFEMReader
{
  public:
  static DataSet<MeshElem<3u, Quad, General>>
  load (const std::string &root_file, const int cycle = 0);
  /// static DataSet<float32, MeshElem<float32, 2u, Quad, General>> load32_2D(const std::string &root_file, const int cycle = 0);
  /// static DataSet<float64, MeshElem<float64, 2u, Quad, General>> load64_2D(const std::string &root_file, const int cycle = 0);

  /// static DataSet<float32, MeshElem<float32, 3u, Tri, General>> load32_tri(const std::string &root_file, const int cycle = 0);
  /// static DataSet<float64, MeshElem<float64, 3u, Tri, General>> load64_tri(const std::string &root_file, const int cycle = 0);
  /// static DataSet<float32, MeshElem<float32, 2u, Tri, General>> load32_2D_tri(const std::string &root_file, const int cycle = 0);
  /// static DataSet<float64, MeshElem<float64, 2u, Tri, General>> load64_2D_tri(const std::string &root_file, const int cycle = 0);
};

} // namespace dray

#endif // DRAY_MFEM_READER_HPP
