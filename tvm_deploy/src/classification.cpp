/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \brief code to deploy image classification models
 * \file classification.cpp
 */
#include "common.hpp"
#include "clipp.hpp"
#include "video.h"
#include <cstdio>
#include <chrono>
#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/c_runtime_api.h>



namespace synset {
// by default class names are empty
static std::vector<std::string> CLASS_NAMES = {};
}  // namespace synset

namespace args {
static std::string video;
static std::string output;
static std::string model_name;
static int gpu = -1;
static int topk = 5;
static bool quiet = false;
static int min_size = 240;
static int crop_size = 224;
}  // namespace args

void ParseArgs(int argc, char** argv) {
    using namespace clipp;

    auto cli = (
        value("video file", args::video),
        value("model name", args::model_name),
        (option("-o", "--output") & value(match::prefix_not("-"), "outfile", args::output)) % "output file, by default no output",
        (option("--gpu") & integer("gpu", args::gpu)) % "Which gpu to use, if -1, means cpu only.",
        (option("--topk") & integer("topk", args::topk)) % "Number of the most probable classes to show as output, by default is 5",
        option("-q", "--quiet").set(args::quiet).doc("quiet mode, no screen output")
    );
    if (!parse(argc, argv, cli) || args::video.empty()) {
        std::cout << make_man_page(cli, argv[0]);
        exit(-1);
    }

    // parse class names
    synset::CLASS_NAMES = LoadClassNames(args::model_name + "_synset.txt");
}

void RunDemo() {
  // tvm module for compiled functions
  tvm::runtime::Module mod_syslib = tvm::runtime::Module::LoadFromFile(args::model_name + "_deploy_lib.so");
  std::ifstream json_in(args::model_name + "_deploy_graph.json", std::ios::in);
  std::string json_data((std::istreambuf_iterator<char>(json_in)), std::istreambuf_iterator<char>());
  json_in.close();

  std::ifstream params_in(args::model_name + "_deploy_0000.params", std::ios::binary);
  std::string params_data((std::istreambuf_iterator<char>(params_in)), std::istreambuf_iterator<char>());
  params_in.close();

  TVMByteArray params_arr;
  params_arr.data = params_data.c_str();
  params_arr.size = params_data.length();

  int dtype_code = kDLFloat;
  int dtype_bits = 32;
  int dtype_lanes = 1;
  int device_type = args::gpu < 0 ? kDLCPU : kDLGPU;
  int device_id = args::gpu < 0 ? 0 : args::gpu;

  tvm::runtime::Module mod = (*tvm::runtime::Registry::Get("tvm.graph_runtime.create"))(json_data, mod_syslib, device_type, device_id);

  // allocate in out buffers
  DLTensor* input;
  int in_ndim = 4;
  int64_t in_shape[4] = {1, 3, args::crop_size, args::crop_size};
  TVMArrayAlloc(in_shape, in_ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &input);
  tvm::runtime::PackedFunc set_input = mod.GetFunction("set_input");
  tvm::runtime::PackedFunc load_params = mod.GetFunction("load_params");
  load_params(params_arr);
  tvm::runtime::PackedFunc run = mod.GetFunction("run");
  DLTensor* y;
  int out_ndim = 2;
  int64_t num_class = synset::CLASS_NAMES.size();
  int64_t out_shape[2] = {1, num_class};
  TVMArrayAlloc(out_shape, out_ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &y);
  tvm::runtime::PackedFunc get_output = mod.GetFunction("get_output");

  // load video data
  video::SetLogging(16);  // no verbose ffmpeg logging
  auto frames = video::ReadFrames(args::video, 25, 320, 240);
  if (!args::quiet) {
      LOG(INFO) << "Read " << frames.size() << " frames.";
  }
  auto n_frames = frames.size();
  std::vector<float> sum_tmp(num_class, 0);
  std::vector<float> output(num_class);

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < frames.size(); ++i) {
      auto im_input = ResizeShortCrop(frames[i], args::min_size, args::crop_size);
      cimg_library::CImg<float> f_input = im_input.normalize(0, 255);
      cimg_forXY(f_input, x, y) {
          float r = f_input(x, y, 0, 0);
          f_input(x, y, 0, 0) = f_input(x, y, 0, 0);
          f_input(x, y, 0, 2) = r;
      }
      // TODO: non-copy version
      TVMArrayCopyFromBytes(input, f_input.data(), args::crop_size*args::crop_size*3*sizeof(float));
      set_input("data", input);
      run();
      get_output(0, y);
      // TODO: non-copy version
      TVMArrayCopyToBytes(y, output.data(), num_class * sizeof(float));
      for (int j = 0; j < num_class; ++j) {
          sum_tmp[j] += output[j];
      }
  }
  // calc mean prediction
  std::transform(sum_tmp.begin(), sum_tmp.end(), output.begin(), [n_frames](float d) -> float { return d / n_frames; });
  
  auto end = std::chrono::steady_clock::now();
  if (!args::quiet) {
    LOG(INFO) << "Elapsed time {Forward->Result}: " << std::chrono::duration<double, std::milli>(end - start).count() << " ms";
  }

  std::vector<int> res_topk = compute::Topk(output, args::topk);

  std::vector<float> res_softmax(num_class);
  compute::Softmax(output, res_softmax);
  std::stringstream ss;
  ss << "The input video is classified to be\n";
  for (int i = 0; i < args::topk; i++) {
      ss <<"\t[" << synset::CLASS_NAMES[res_topk[i]] <<  "], with probability " << std::fixed << std::setprecision(3) << res_softmax[res_topk[i]] << "\n";
  }

  if (!args::quiet) {
      LOG(INFO) << ss.str();
  }

  // output file
  if (!args::output.empty()) {
      std::ofstream outfile(args::output);
      outfile << ss.str();
      outfile.close();
  }

  TVMArrayFree(input);
  TVMArrayFree(y);
}

int main(int argc, char** argv) {
    ParseArgs(argc, argv);
    RunDemo();
    return 0;
}
