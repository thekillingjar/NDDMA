/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_NODE_CONVERTER_AICPU_FFTS_NODE_CONVERTER_H
#define AIR_CXX_RUNTIME_V2_NODE_CONVERTER_AICPU_FFTS_NODE_CONVERTER_H
#include "graph/node.h"
#include "register/node_converter_registry.h"
#include "register/ffts_node_converter_registry.h"
#include "register/ffts_node_calculater_registry.h"

namespace gert {
constexpr size_t MAX_THREAD_DIM = 32UL;
enum class MemPoolType {
  kFirstMemPool,
  kSecondaryMemPool
};

enum class SliceShapeIndex {
  kNotLastInShapes,
  kLastInShapes,
  kNotLastOutShapes = 4,
  kLastOutShapes = 5,
  kTotalNum = 8
};

struct ThreadInfo {
  bg::ValueHolderPtr thread_para;
  std::vector<bg::ValueHolderPtr> sgt_thread_info;
};

struct RefNodeInfo {
  uint32_t out_index;
  ge::NodePtr node;
  std::map<size_t, size_t> ref_map;
};

ge::graphStatus CalcAICpuCCArgsMem(const ge::NodePtr &node, const LoweringGlobalData *global_data, size_t &total_size,
                                   size_t &pre_data_size, std::unique_ptr<uint8_t[]> &pre_data_ptr);
ge::graphStatus CalcAICpuTfArgsMem(const ge::NodePtr &node, const LoweringGlobalData *global_data, size_t &total_size,
                                   size_t &pre_data_size, std::unique_ptr<uint8_t[]> &pre_data_ptr);
LowerResult LoweringFFTSAiCpuNode(const ge::NodePtr &node, const FFTSLowerInput &lower_input);
}
#endif  // AIR_CXX_RUNTIME_V2_NODE_CONVERTER_AICPU_FFTS_NODE_CONVERTER_H
