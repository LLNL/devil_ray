#ifndef DRAY_BVH_WRITER_HPP
#define DRAY_BVH_WRITER_HPP

#include <dray/array.hpp>
#include <dray/ray.hpp>
#include <dray/types.hpp>
#include <dray/exports.hpp>
#include <dray/linear_bvh_builder.hpp>

#include <fstream>  // for std::ofstream
#include <sstream>  // for std::ostringstream

// Stolen from Axom: Thanks George!
namespace dray
{

//------------------------------------------------------------------------------
template < typename FloatType >
void write_box3d( const FloatType& xmin,
                  const FloatType& ymin,
                  const FloatType& zmin,
                  const FloatType& xmax,
                  const FloatType& ymax,
                  const FloatType& zmax,
                  int32& numPoints,
                  int32& numBins,
                  std::ostringstream& nodes,
                  std::ostringstream& cells  )
{
  nodes << xmin << " " << ymin << " " << zmin << std::endl;
  nodes << xmax << " " << ymin << " " << zmin << std::endl;
  nodes << xmax << " " << ymax << " " << zmin << std::endl;
  nodes << xmin << " " << ymax << " " << zmin << std::endl;

  nodes << xmin << " " << ymin << " " << zmax << std::endl;
  nodes << xmax << " " << ymin << " " << zmax << std::endl;
  nodes << xmax << " " << ymax << " " << zmax << std::endl;
  nodes << xmin << " " << ymax << " " << zmax << std::endl;

  constexpr int32 NUM_NODES = 8;
  int32 offset = numPoints;
  cells << NUM_NODES << " ";
  for ( int32 i=0; i < NUM_NODES; ++i )
  {
    cells << (offset + i) << " ";
  }
  cells << "\n";

  numBins   += 1;
  numPoints += NUM_NODES;
}

//------------------------------------------------------------------------------
template < typename FloatType>
void write_leftbox( const Vec< FloatType, 4 >& first,
                    const Vec< FloatType, 4 >& second,
                    int32& numPoints,
                    int32& numBins,
                    std::ostringstream& nodes,
                    std::ostringstream& cells  )
{
  const FloatType& xmin = first[ 0 ];
  const FloatType& ymin = first[ 1 ];

  const FloatType& xmax = first[ 3 ];
  const FloatType& ymax = second[ 0 ];

  const FloatType& zmin = first[ 2 ];
  const FloatType& zmax = second[ 2 ];

  write_box3d( xmin, ymin, zmin,
               xmax, ymax, zmax,
               numPoints, numBins, nodes, cells );

}

//------------------------------------------------------------------------------
template < typename FloatType >
void write_righbox( const Vec< FloatType, 4 >& second,
                    const Vec< FloatType, 4 >& third,
                    int32& numPoints,
                    int32& numBins,
                    std::ostringstream& nodes,
                    std::ostringstream& cells  )
{
  const FloatType& xmin = second[ 2 ];
  const FloatType& ymin = second[ 3 ];

  const FloatType& xmax = third[ 1 ];
  const FloatType& ymax = third[ 2 ];

  const FloatType& zmin = third[ 0 ];
  const FloatType& zmax = third[ 3 ];

  write_box3d( xmin, ymin, zmin,
               xmax, ymax, zmax,
               numPoints, numBins, nodes, cells );

}

void write_root( const AABB<>& root,
                 int32& numPoints,
                 int32& numBins,
                 std::ostringstream& nodes,
                 std::ostringstream& cells,
                 std::ostringstream& levels,
                 std::ostringstream& leafs )
{

  write_box3d( root.m_ranges[0].min(), root.m_ranges[1].min(), root.m_ranges[2].min(),
               root.m_ranges[0].max(), root.m_ranges[1].max(), root.m_ranges[2].max(),
               numPoints,
               numBins,
               nodes,
               cells  );
  levels << "0\n";
  leafs << 0 << std::endl;
}

//------------------------------------------------------------------------------
template < typename FloatType>
void write_recursive( const Vec< FloatType, 4 >* inner_nodes,
                      int32 current_node,
                      int32 level,
                      int32& numPoints,
                      int32& numBins,
                      std::ostringstream& nodes,
                      std::ostringstream& cells,
                      std::ostringstream& levels,
                      std::ostringstream& leafs)
{
  // STEP 0: get the flat BVH layout
  const Vec< FloatType, 4> first4  = inner_nodes[current_node + 0];
  const Vec< FloatType, 4> second4 = inner_nodes[current_node + 1];
  const Vec< FloatType, 4> third4  = inner_nodes[current_node + 2];

  // STEP 1: extract children information
  int32 l_child;
  int32 r_child;
  constexpr int32 isize = sizeof(int32);
  Vec< FloatType, 4 > children = inner_nodes[current_node + 3];
  memcpy(&l_child,&children[0],isize);
  memcpy(&r_child,&children[1],isize);

  write_leftbox< FloatType>( first4, second4, numPoints, numBins,
                             nodes, cells );
  levels << level << std::endl;
  leafs << (int32) (l_child < 0) << std::endl;
  write_righbox< FloatType >( second4, third4, numPoints, numBins,
                              nodes, cells );
  levels << level << std::endl;
  leafs << (int32) (r_child < 0) << std::endl;

  // STEP 2: check left
  if ( l_child > - 1 )
  {
    write_recursive< FloatType>(
        inner_nodes, l_child, level+1, numPoints, numBins,
        nodes, cells, levels, leafs );

  }

  // STEP 3: check right
  if ( r_child > -1 )
  {
    write_recursive< FloatType>(
        inner_nodes, r_child, level+1, numPoints, numBins,
        nodes, cells, levels, leafs );

  }

}
//------------------------------------------------------------------------------
void writeVtkFile(const BVH &bvh, const std::string& fileName )
{
  const Vec<float32,4> * inner_nodes = bvh.m_inner_nodes.get_host_ptr_const();

  std::ostringstream nodes;
  std::ostringstream cells;
  std::ostringstream levels;
  std::ostringstream leafs;

  // STEP 0: Write VTK header
  std::ofstream ofs;
  ofs.open( fileName.c_str() );
  ofs << "# vtk DataFile Version 3.0\n";
  ofs << " BVHTree \n";
  ofs << "ASCII\n";
  ofs << "DATASET UNSTRUCTURED_GRID\n";

  // STEP 1: write root
  int32 numPoints = 0;
  int32 numBins   = 0;
  write_root(bvh.m_bounds, numPoints, numBins,nodes,cells,levels,leafs);


  // STEP 2: traverse the BVH and dump each bin
  constexpr int32 ROOT = 0;
  write_recursive(
      inner_nodes, ROOT, 1, numPoints, numBins, nodes, cells, levels, leafs );

  // STEP 3: write nodes
  ofs << "POINTS " << numPoints << " double\n";
  ofs << nodes.str() << std::endl;

  // STEP 4: write cells
  const int32 nnodes = 8;
  ofs << "CELLS " << numBins << " " << numBins*(nnodes+1) << std::endl;
  ofs << cells.str() << std::endl;

  // STEP 5: write cell types
  ofs << "CELL_TYPES " << numBins << std::endl;
  const int32 cellType = 12;
  for ( int32 i=0; i < numBins; ++i )
  {
    ofs << cellType << std::endl;
  }

  // STEP 6: dump level information
  ofs << "CELL_DATA " << numBins << std::endl;
  ofs << "SCALARS level int\n";
  ofs << "LOOKUP_TABLE default\n";
  ofs << levels.str() << std::endl;

  ofs << "SCALARS leafs int\n";
  ofs << "LOOKUP_TABLE default\n";
  ofs << leafs.str() << std::endl;
  ofs << std::endl;
  

  // STEP 7: close file
  ofs.close();
}
}; // namespace dray

#endif
