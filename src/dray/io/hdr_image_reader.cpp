#include <dray/io/hdr_image_reader.hpp>
#include <dray/error.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

namespace dray
{



Array<Vec<float32,3>> read_hdr_image(const std::string filename, int &width, int &height)
{
  int n;
  float *rgb = stbi_loadf(filename.c_str(), &width, &height, &n, 0);

  if(!rgb)
  {
    DRAY_ERROR("Failed to load hrd file '"<<filename<<"'");

  }
  if (n != 3)
  {
    DRAY_ERROR("HDR loade: expect 3 components got "<<n);
  }
  Array<Vec<float32,3>> image;
  image.resize(width*height);

  Vec<float32,3> *image_ptr = image.get_host_ptr();

  for(int32 i = 0; i < width * height; ++i)
  {
    int32 offset = i * 3;
    Vec<float32,3> color;
    color[0] = rgb[offset + 0];
    color[1] = rgb[offset + 1];
    color[2] = rgb[offset + 2];
    image_ptr[i] = color;
  }

  free(rgb);
  return image;
}

} // namespace dray

