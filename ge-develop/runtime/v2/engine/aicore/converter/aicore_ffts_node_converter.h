/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_NODE_CONVERTER_AICORE_FFTS_NODE_CONVERTER_H_
#define AIR_CXX_RUNTIME_V2_NODE_CONVERTER_AICORE_FFTS_NODE_CONVERTER_H_
#include "graph/node.h"
#include "register/node_converter_registry.h"
#include "register/ffts_node_converter_registry.h"
#include "register/ffts_node_calculater_registry.h"
namespace gert {
struct AtomicFFTSLowerArg {
  std::vector<bg::ValueHolderPtr> &tiling_ret;
  bg::DevMemValueHolderPtr &workspaces_addrs;
  std::vector<bg::ValueHolderPtr> &output_sizes;
  std::vector<bg::ValueHolderPtr> thread_ret;
  std::vector<bg::DevMemValueHolderPtr> &output_addrs;
  bg::ValueHolderPtr out_mem_type;
  std::vector<bg::ValueHolderPtr> &launch_arg;
  bg::ValueHolderPtr thread_para;
};

bg::ValueHolderPtr LaunchFFTSAtomicClean(const ge::NodePtr &node, const FFTSLowerInput &lower_input,
                                         const AtomicFFTSLowerArg &lower_args);
}
#endif  // AIR_CXX_RUNTIME_V2_NODE_CONVERTER_AICORE_FFTS_NODE_CONVERTER_H_
