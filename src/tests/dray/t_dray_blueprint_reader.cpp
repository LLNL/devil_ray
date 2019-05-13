#include "gtest/gtest.h"
#include "test_config.h"
#include <dray/camera.hpp>
#include <dray/mfem_volume_integrator.hpp>
#include <dray/utils/timer.hpp>
#include <dray/utils/data_logger.hpp>
#include <dray/shaders.hpp>
#include <dray/utils/png_encoder.hpp>

#include <dray/mfem_data_set.hpp>
#include <mfem/fem/conduitdatacollection.hpp>

#include <fstream>
#include <stdlib.h>
#include <sstream>

#include <mfem.hpp>
using namespace mfem;

TEST(dray_mfem_blueprint, dray_mfem_blueprint)
{
  //std::string file_name = std::string(DATA_DIR) + "results/Laghos";
  std::string file_name = "/usr/workspace/wsb/larsen30/pascal/alpine/mfem_ascent/ascent/build/examples/proxies/laghos/small_run/Laghos";
  std::cout<<"File name "<<file_name<<"\n";
  
  dray::ColorTable color_table("cool2warm");
  color_table.add_alpha(0.f,  0.01f);
  color_table.add_alpha(0.1f, 0.09f);
  color_table.add_alpha(0.2f, 0.01f);
  color_table.add_alpha(0.3f, 0.09f);
  color_table.add_alpha(0.4f, 0.01f);
  color_table.add_alpha(0.5f, 0.01f);
  color_table.add_alpha(0.6f, 0.01f);
  color_table.add_alpha(0.7f, 0.09f);
  color_table.add_alpha(0.8f, 0.01f);
  color_table.add_alpha(0.9f, 0.01f);
  color_table.add_alpha(1.0f, 0.0f);

  mfem::ConduitDataCollection col(file_name);
  col.SetProtocol("conduit_json");
  int count = 0; 
  for(int cycle = 0; cycle <= 7465; cycle += 5)
  //for(int cycle = 20; cycle <= 20; cycle += 5)
  {
    col.Load(cycle);

    dray::MFEMDataSet data_set;
    data_set.set_mesh(col.GetMesh());

    auto field_map = col.GetFieldMap(); 
    for(auto it = field_map.begin(); it != field_map.end(); ++it)
    {
      data_set.add_field(it->second, it->first);
    }
    
    data_set.print_self();

    //------- DRAY CODE --------

    // Volume rendering.
    dray::Camera camera;
    camera.set_width(1024);
    camera.set_height(1024);
    //camera.set_width(500);
    //camera.set_height(500);
    camera.reset_to_bounds(data_set.get_mesh().get_bounds());


     dray::Vec<dray::float32,3> pos;
     pos[0] = 4.0;
     pos[1] = 3.5;
     pos[2] = 7.5;
     camera.set_pos(pos);

    dray::ray32 rays;
    camera.create_rays(rays);
    dray::MFEMVolumeIntegrator integrator(data_set.get_mesh(), data_set.get_field("Density"));
    //dray::MFEMVolumeIntegrator integrator(data_set.get_mesh(), data_set.get_field("Specific Internal Energy"));
    dray::Shader::set_color_table(color_table);
    dray::Array<dray::Vec<dray::float32,4>> color_buffer = integrator.integrate(rays);

    dray::PNGEncoder png_encoder;
    png_encoder.encode( (float *) color_buffer.get_host_ptr(), camera.get_width(), camera.get_height() );
    std::stringstream ss;
    ss<<"laghos_"<<count<<".png";
    png_encoder.save(ss.str());
    count++;
  }
  DRAY_LOG_WRITE("mfem");
}
