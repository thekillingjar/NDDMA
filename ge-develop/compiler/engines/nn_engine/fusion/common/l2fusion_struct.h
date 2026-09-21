/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSION_ENGINE_FUSION_COMMON_L2FUSION_STRUCT_H_
#define FUSION_ENGINE_FUSION_COMMON_L2FUSION_STRUCT_H_

#include <map>
#include <string>
#include "runtime/kernel.h"

namespace fe {
struct L2Data {
  uint32_t l2Index;
  uint64_t l2Addr;
  uint64_t l2PageNum;
};

using L2DataMap = std::map<uint64_t, L2Data>;   // the key is ddr addr
using L2DataPair = std::pair<uint64_t, L2Data>;  // the key is ddr addr

struct TaskL2Info {
  std::string nodeName;
  rtL2Ctrl_t l2ctrl;

  L2DataMap input;
  L2DataMap output;
  uint32_t isUsed;
};

using TaskL2InfoMap = std::map<std::string, TaskL2Info>;    // the key is nodeName
using TaskL2InfoPair = std::pair<std::string, TaskL2Info>;  // the key is nodeName
}  // namespace fe
#endif  // FUSION_ENGINE_FUSION_COMMON_L2FUSION_STRUCT_H_
