// Copyright 2019 Lawrence Livermore National Security, LLC and other
// Devil Ray Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "test_config.h"
#include "gtest/gtest.h"

#include "t_utils.hpp"
#include <dray/io/blueprint_reader.hpp>
#include <dray/io/blueprint_low_order.hpp>
#include <dray/rendering/sampling.hpp>
#include <dray/rendering/disney_sampling.hpp>
#include <dray/random.hpp>


#include <dray/math.hpp>

#include <conduit.hpp>
#include <conduit_relay.hpp>
#include <conduit_blueprint.hpp>


#include <fstream>
#include <stdlib.h>

using namespace dray;

void write_vectors(std::vector<Vec<float32,3>> &dirs, std::string name);

TEST (dray_test_sampling, dray_disney)
{
  Array<Vec<uint32,2>> rstate;
  rstate.resize(1);
  bool deterministic = true;
  seed_rng(rstate, deterministic);
  Vec<uint32,2> rand_state = rstate.get_value(0);

  int32 samples = 10;

  Vec<float32,3> normal = {{0.f, 1.f, 0.f}};
  Vec<float32,3> view = {{0.1f, .2f, 0.5f}};
  view.normalize();

  Material mat;
  mat.m_ior = 1.1;
  mat.m_spec_trans = 0.5f;
  mat.m_specular = 0.5f;
  mat.m_roughness = 0.08f;
  std::vector<Vec<float32,3>> dirs;
  for(int i = 0; i < samples; ++i)
  {
    bool specular;
    Vec<float32,3> new_dir =  sample_disney(view, normal, mat, specular, rand_state);
    new_dir.normalize();
    dirs.push_back(new_dir);
  }
  dirs.push_back(normal * 2.f);
  dirs.push_back(view* 3.f);
  write_vectors(dirs,"sample_disney");
}

TEST (dray_test_sampling, dray_ggx)
{
  Array<Vec<uint32,2>> rstate;
  rstate.resize(1);
  bool deterministic = true;
  seed_rng(rstate, deterministic);
  Vec<uint32,2> rand_state = rstate.get_value(0);

  int32 samples = 100;

  Vec<float32,3> normal = {{0.f, 1.f, 0.f}};

  std::vector<Vec<float32,3>> dirs;
  for(int i = 0; i < samples; ++i)
  {
    Vec<float32,2> rand;
    rand[0] = randomf(rand_state);
    rand[1] = randomf(rand_state);
    Vec<float32,3> new_dir = sample_ggx(0.f, rand);
    new_dir.normalize();
    dirs.push_back(new_dir);
  }
  write_vectors(dirs,"ggx");
}

TEST (dray_test_sampling, dray_cos)
{
  Array<Vec<uint32,2>> rstate;
  rstate.resize(1);
  bool deterministic = true;
  seed_rng(rstate, deterministic);
  Vec<uint32,2> rand_state = rstate.get_value(0);

  int32 samples = 100;

  Vec<float32,3> normal = {{0.f, 1.f, 0.f}};

  std::vector<Vec<float32,3>> dirs;
  for(int i = 0; i < samples; ++i)
  {
    Vec<float32,2> rand;
    rand[0] = randomf(rand_state);
    rand[1] = randomf(rand_state);
    Vec<float32,3> new_dir = cosine_weighted_hemisphere (normal, rand);
    new_dir.normalize();
    dirs.push_back(new_dir);
  }
  write_vectors(dirs,"cosine_weighted");
}

TEST (dray_test_sampling, dray_spec)
{
  Array<Vec<uint32,2>> rstate;
  rstate.resize(1);
  bool deterministic = true;
  seed_rng(rstate, deterministic);
  Vec<uint32,2> rand_state = rstate.get_value(0);

  int32 samples = 100;
  Vec<float32,3> normal = {{-0.296209, 0, -0.955123}};
  Vec<float32,3> view = {{-0.0175326, 0.107532, -0.994047}};
  //Vec<float32,3> normal = {{0.f, 1.f, 0.f}};
  //Vec<float32,3> view = {{0.8f, .1f, 0.f}};
  view.normalize();

  const float32 roughness = 0.005f;

  std::vector<Vec<float32,3>> dirs;
  for(int i = 0; i < samples; ++i)
  {
    Vec<float32,2> rand;
    rand[0] = randomf(rand_state);
    rand[1] = randomf(rand_state);
    Vec<float32,3> new_dir = specular_sample(normal, view, rand, roughness, true);
    Vec<float32,3> half = new_dir + view;
    half.normalize();
    float32 pdf = eval_pdf(new_dir,view,normal,roughness,0.f);
    //dirs.push_back(half*3.f);
    dirs.push_back(new_dir);
  }

  dirs.push_back(normal * 2.f);
  dirs.push_back(view* 3.f);

  write_vectors(dirs,"specular");
}

void write_vectors(std::vector<Vec<float32,3>> &dirs, std::string name)
{
  std::vector<float> x;
  std::vector<float> y;
  std::vector<float> z;
  std::vector<int32> conn;


  x.push_back(0.f);
  y.push_back(0.f);
  z.push_back(0.f);

  int conn_count = 1;

  for(auto d : dirs)
  {
    x.push_back(d[0]);
    y.push_back(d[1]);
    z.push_back(d[2]);
    conn.push_back(0);
    conn.push_back(conn_count);
    conn_count++;
  }

  conduit::Node domain;

  domain["coordsets/coords/type"] = "explicit";
  domain["coordsets/coords/values/x"].set(x);
  domain["coordsets/coords/values/y"].set(y);
  domain["coordsets/coords/values/z"].set(z);
  domain["topologies/mesh/type"] = "unstructured";
  domain["topologies/mesh/coordset"] = "coords";
  domain["topologies/mesh/elements/shape"] = "line";
  domain["topologies/mesh/elements/connectivity"].set(conn);

  conduit::Node dataset;
  dataset.append() = domain;
  conduit::Node info;
  if(!conduit::blueprint::mesh::verify(dataset,info))
  {
    info.print();
  }
  conduit::relay::io_blueprint::save(domain, name+".blueprint_root");
}

