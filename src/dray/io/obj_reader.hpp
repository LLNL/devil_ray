#ifndef OBJ_LOADER_H
#define OBJ_LOADER_H

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

void read_obj(const std::string file_name, 
              dray::Array<dray::float32> &a_verts, 
              dray::Array<dray::int32> &a_indices)
{


  tinyobj::attrib_t attrib;
  std::vector<tinyobj::shape_t> shapes;
  std::vector<tinyobj::material_t> materials;
    
  std::string err;

  bool ret = tinyobj::LoadObj(&attrib,
                              &shapes, 
                              &materials, 
                              &err, 
                              file_name.c_str());

  if (!err.empty()) 
  { // `err` may contain warning message.
   std::cerr << err << std::endl;
  }
  
  if (!ret) 
  {
    exit(1);
  }

  a_verts.set(&attrib.vertices[0], attrib.vertices.size()); 
  
  // count the number of triangles
  int tris = 0;
  for (size_t s = 0; s < shapes.size(); s++) 
  {
    tris += shapes[s].mesh.num_face_vertices.size();
  }
  a_indices.resize(tris * 3); 
  int tot = tris*3; 
  dray::int32 *indices = a_indices.get_host_ptr();

  int indices_offset = 0;
  // Loop over shapes
  for (size_t s = 0; s < shapes.size(); s++) 
  {
    // Loop over faces(polygon) defults to triangulate
    size_t index_offset = 0;
    for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) 
    {
      int fv = shapes[s].mesh.num_face_vertices[f];
      // Loop over vertices in the face.
      for (size_t v = 0; v < fv; v++) 
      {
        // access to vertex
        tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];
        indices[indices_offset + 0] =  idx.vertex_index;
        indices_offset++;
        //tinyobj::real_t vx = attrib.vertices[3*idx.vertex_index+0];
        //tinyobj::real_t vy = attrib.vertices[3*idx.vertex_index+1];
        //tinyobj::real_t vz = attrib.vertices[3*idx.vertex_index+2];
        
        //tinyobj::real_t nx = attrib.normals[3*idx.normal_index+0];
        //tinyobj::real_t ny = attrib.normals[3*idx.normal_index+1];
        //tinyobj::real_t nz = attrib.normals[3*idx.normal_index+2];

        //tinyobj::real_t tx = attrib.texcoords[2*idx.texcoord_index+0];
        //tinyobj::real_t ty = attrib.texcoords[2*idx.texcoord_index+1];
        // Optional: vertex colors
        // tinyobj::real_t red = attrib.colors[3*idx.vertex_index+0];
        // tinyobj::real_t green = attrib.colors[3*idx.vertex_index+1];
        // tinyobj::real_t blue = attrib.colors[3*idx.vertex_index+2];
      }
      index_offset += fv;
  
      // per-face material
      //shapes[s].mesh.material_ids[f];
    } 
  }
}

#endif
