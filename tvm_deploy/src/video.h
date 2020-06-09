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
 * \file common.hpp
 * \brief Common functions for GluonCV cpp inference demo
 * \author
 */
#ifndef GCV_TVM_VIDEO_H_
#define GCV_TVM_VIDEO_H_
#define cimg_display 0
#include "CImg.h"
#include <string>
#include <vector>

namespace video {
std::vector<cimg_library::CImg<float> > ReadFrames(std::string video_name, int interval = 25);
}
#endif
