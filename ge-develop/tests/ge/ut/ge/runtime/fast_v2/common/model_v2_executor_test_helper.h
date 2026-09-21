/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_TESTS_UT_GE_RUNTIME_V2_COMMON_MODEL_V2_EXECUTOR_TEST_HELPER_H_
#define AIR_CXX_TESTS_UT_GE_RUNTIME_V2_COMMON_MODEL_V2_EXECUTOR_TEST_HELPER_H_
#include "runtime/model_v2_executor.h"
#include "core/execution_data.h"
#include "exe_graph/runtime/extended_kernel_context.h"
namespace gert {
class ModelV2ExecutorTestHelper {
 public:
  static void SetExecutionData(ExecutionData *execution_data, SubExeGraphType graph_type, ModelV2Executor *executor) {
    executor->graphs_[graph_type].execution_data_ = execution_data;
  }
  static void *GetExecutionData(ModelV2Executor *executor, SubExeGraphType graph_type) {
    return executor->graphs_[graph_type].execution_data_;
  }
  static Node *GetNodeByKernelName(void *execution_data, const char *kernel_name) {
    auto edata = reinterpret_cast<ExecutionData *>(execution_data);
    for (size_t i = 0; i < edata->base_ed.node_num; ++i) {
      auto context = reinterpret_cast<ExtendedKernelContext *>(&edata->base_ed.nodes[i]->context);
      if (strcmp(context->GetKernelName(), kernel_name) == 0) {
        return edata->base_ed.nodes[i];
      }
    }
    return nullptr;
  }
  static std::vector<Node *> GetNodesByKernelType(ModelV2Executor *executor, const char *kernel_type) {
    std::vector<Node *> nodes;
    for (int32_t i = 0; i < kSubExeGraphTypeEnd; ++i) {
      auto execution_data = GetExecutionData(executor, static_cast<SubExeGraphType>(i));
      if (execution_data == nullptr) {
        continue;
      }
      auto tmp_nodes = GetNodesByKernelType(execution_data, kernel_type);
      nodes.insert(nodes.end(), tmp_nodes.begin(), tmp_nodes.end());
    }
    return nodes;
  }
  static std::vector<Node *> GetNodesByKernelType(void *execution_data, const char *kernel_type) {
    std::vector<Node *> nodes;
    auto edata = reinterpret_cast<ExecutionData *>(execution_data);
    for (size_t i = 0; i < edata->base_ed.node_num; ++i) {
      auto context = reinterpret_cast<ExtendedKernelContext *>(&edata->base_ed.nodes[i]->context);
      if (strcmp(context->GetKernelType(), kernel_type) == 0) {
        nodes.push_back(edata->base_ed.nodes[i]);
      }
    }
    return nodes;
  }
  static Chain *GetOutChain(Node *node, size_t out_index) {
    auto context = reinterpret_cast<KernelContext *>(&node->context);
    return context->GetOutput(out_index);
  }
  static Chain *GetOutChain(void *execution_data, const char *kernel_name, size_t out_index) {
    auto node = GetNodeByKernelName(execution_data, kernel_name);
    if (node == nullptr) {
      return nullptr;
    }
    return GetOutChain(node, out_index);
  }

  static ge::graphStatus CheckIoReuseAddrs(ModelV2Executor *executor, Tensor **inputs, size_t input_num,
                                           Tensor **outputs, size_t output_num) {
    return executor->CheckIoReuseAddrs(inputs, input_num, outputs, output_num);
  }
};
}  // namespace gert

#endif  // AIR_CXX_TESTS_UT_GE_RUNTIME_V2_COMMON_MODEL_V2_EXECUTOR_TEST_HELPER_H_
