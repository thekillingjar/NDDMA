/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/ge_error_codes.h"
#include "exe_graph/runtime/storage_shape.h"
#include "exe_graph/runtime/tensor.h"
#include "register/node_converter_registry.h"
#include "register/kernel_registry.h"
#include "core/utils/tensor_utils.h"
#include "exe_graph/runtime/gert_tensor_data.h"

namespace gert {
namespace {
const std::string kLowerPrefix = "Lower_";
const char *kStubLowerFunc = "_stub_lower_func";
REGISTER_NODE_CONVERTER(kStubLowerFunc, [](const ge::NodePtr &node, const LowerInput &lower_input) {
  LowerResult ret;
  auto merged_holders = lower_input.input_shapes;
  merged_holders.insert(merged_holders.end(), lower_input.input_addrs.begin(), lower_input.input_addrs.end());

  int64_t logic_stream_id = node->GetOpDesc()->GetStreamId();
  auto stream_id_holder = bg::ValueHolder::CreateConst(&logic_stream_id, sizeof(logic_stream_id));
  merged_holders.emplace_back(stream_id_holder); // make stream id as last input

  if (node->GetAllOutDataAnchorsSize() == 0U) {
    ret.order_holders.push_back(bg::ValueHolder::CreateVoid<bg::ValueHolder>((kLowerPrefix + node->GetType()).c_str(), merged_holders));
  } else {
    auto holders = bg::DevMemValueHolder::CreateDataOutput((kLowerPrefix + node->GetType()).c_str(), merged_holders,
                                                           node->GetAllOutDataAnchorsSize() << 1U,
                                                           node->GetOpDesc()->GetStreamId());
    ret.out_shapes.insert(ret.out_shapes.end(), holders.begin(), holders.begin() + node->GetAllOutDataAnchorsSize());
    ret.out_addrs.insert(ret.out_addrs.end(), holders.begin() + node->GetAllOutDataAnchorsSize(), holders.end());
  }
  return ret;
});
}  // namespace

bool MockLessImportantNodeConverter(const ge::NodePtr &node, bool override) {
  if (!node->GetOpDesc()->HasAttr("_ge_attr_lowering_func")) {
    if (NodeConverterRegistry::GetInstance().FindNodeConverter(node->GetType()) == nullptr || override) {
      return ge::AttrUtils::SetStr(node->GetOpDesc(), "_ge_attr_lowering_func", kStubLowerFunc);
    }
    return false;
  }
  return false;
}

void MockLessImportantNodeConverter(const ge::ComputeGraphPtr &graph, const std::set<std::string> &exclude_ops,
                                    const std::set<std::string> &exclude_nodes, bool override) {
  for (auto &node : graph->GetAllNodes()) {
    if (exclude_ops.count(node->GetType()) || exclude_nodes.count(node->GetName())) {
      continue;
    }
    MockLessImportantNodeConverter(node, override);
  }
}

void MockLessImportantNodeKernel(const ge::NodePtr &node) {
  if (!MockLessImportantNodeConverter(node, false)) {
    return;
  }
  KernelRegistry::KernelFuncs funcs = {};
  funcs.run_func = [](KernelContext *context) {
    auto stream_id_input_idx = context->GetInputNum() - 1;
    auto stream_id = context->GetInputValue<int64_t>(stream_id_input_idx);
    size_t num_outputs = context->GetOutputNum();
    for (size_t i = (num_outputs >> 1U); i < num_outputs; i++) {
      const static char kBuffer[1024] = {};
      auto gtd = context->GetOutputPointer<GertTensorData>(i);
      *gtd = {const_cast<char *>(kBuffer), sizeof(kBuffer), kOnDeviceHbm, stream_id};
    }
    return ge::GRAPH_SUCCESS;
  };
  funcs.outputs_creator = [](const ge::FastNode *n, KernelContext *context) {
    size_t num_outputs = (context->GetOutputNum() >> 1U);
    for (size_t i = 0U; i < num_outputs; i++) {
      context->GetOutput(i)->SetWithDefaultDeleter(new StorageShape());
      context->GetOutput(i + num_outputs)->SetWithDefaultDeleter(new GertTensorData());
    }
    return ge::GRAPH_SUCCESS;
  };
  KernelRegistry::GetInstance().RegisterKernel(kLowerPrefix + node->GetType(), {funcs, ""});
}

void MockLessImportantNodeKernel(const ge::ComputeGraphPtr &graph, const std::set<std::string> &exclude_ops,
                                 const std::set<std::string> &exclude_nodes) {
  for (auto &node : graph->GetAllNodes()) {
    if (exclude_ops.count(node->GetType()) || exclude_nodes.count(node->GetName())) {
      continue;
    }
    MockLessImportantNodeKernel(node);
  }
}
}  // namespace gert
