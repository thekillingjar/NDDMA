/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GRADIENT_SPLIT_TUNE_H
#define GRADIENT_SPLIT_TUNE_H

#include <string>
#include "nlohmann/json.hpp"

struct BPTimeInfo {
  std::string opName;
  uint64_t time;
};

struct GradientNode {
  std::string traceNodeName;
  std::string groupName;
  std::string dataType;
  uint64_t gradientSize;
  uint32_t graphId;
};

struct GradientInfo {
  std::string opName;
  std::string groupName;
  std::string dataType;
  uint64_t time;
  uint64_t gradientSize;
  uint32_t graphId;
  uint32_t index;
};

using IndexInfo = struct {
  int gradientSizeIndex;
  int dataTypeIndex;
  int graphIdIndex;
  int traceNodeNameIndex;
  int groupNameIndex;
};

#endif