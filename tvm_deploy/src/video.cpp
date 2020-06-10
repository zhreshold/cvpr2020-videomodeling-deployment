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
 *  Copyright (c) 2020 by Contributors
 * \file video.cpp
 * \brief simple video reading functions
 * \author
 */
#include "video.h"
#include <decord/video_interface.h>
#include <decord/runtime/registry.h>

namespace video {
void SetLogging(int log_level) {
  (*decord::runtime::Registry::Get("logging._CAPI_SetLoggingLevel"))(log_level);
}

std::vector<cimg_library::CImg<uint8_t> > ReadFrames(std::string video_name, int interval,
                                                   int width, int height) {
    auto reader = decord::GetVideoReader(video_name, decord::kCPU, width, height, 0);
    auto num_frames = reader->GetFrameCount();
    std::vector<int64_t> indices;
    for (int i = 0; i < num_frames - 1; ++i) {
      if (i % interval == 0) {
        indices.emplace_back(i);
      }
    }
    decord::runtime::NDArray buf;
    auto frames = reader->GetBatch(indices, buf);
    auto selected = indices.size();
    auto stride = 3 * width * height;
    CHECK_EQ(frames.Size(), selected * stride);
    std::vector<cimg_library::CImg<uint8_t> > out;
    out.reserve(selected);
    std::vector<uint8_t> buffer;
    frames.CopyTo(buffer);
    for (int i = 0; i < selected; ++i) {
      cimg_library::CImg<uint8_t> src(buffer.data() + stride * i, 
                                    3, width, height, 1);
      out.emplace_back(src.permute_axes("yzcx"));
    }
    return out;
}
}
