/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "faker/nodes_faker_for_exe.h"
#include <vector>
#include <map>
#include <memory>
#include <string>
#include "register/node_converter_registry.h"
#include "common/checker.h"
#include "stub/kernel_stub.h"
#include "stub/gert_runtime_stub.h"

// fake ops
#include "fake_shape.h"
namespace gert {
namespace {
LowerResult FakedConverter(const ge::NodePtr &node, const LowerInput &lower_input) {
  auto faker = NodesFakerForExe::GetNodeExeFaker(node->GetType());
  if (faker == nullptr) {
    return {HyperStatus::ErrorStatus("Failed to lowering node %s(%s), does not support now", node->GetName().c_str(),
                                     node->GetType().c_str()),
            {},
            {},
            {}};
  }

  auto output_size = node->GetAllOutDataAnchorsSize();
  std::vector<bg::ValueHolderPtr> kernel_inputs;
  kernel_inputs.push_back(bg::ValueHolder::CreateConst(node->GetType().c_str(), node->GetType().size() + 1, true));
  kernel_inputs.push_back(lower_input.global_data->GetStreamById(node->GetOpDesc()->GetStreamId()));
  kernel_inputs.push_back(lower_input.global_data->GetOrCreateL2Allocator(
      node->GetOpDesc()->GetStreamId(), {faker->GetOutPlacement(), AllocatorUsage::kAllocNodeOutput}));
  kernel_inputs.insert(kernel_inputs.cend(), lower_input.input_shapes.begin(), lower_input.input_shapes.end());
  kernel_inputs.insert(kernel_inputs.cend(), lower_input.input_addrs.begin(), lower_input.input_addrs.end());

  auto outputs = bg::DevMemValueHolder::CreateDataOutput("FakeExecuteNode", kernel_inputs, output_size * 2,
                                                         node->GetOpDesc()->GetStreamId());

  for (size_t i = output_size; i < outputs.size(); ++i) {
    bg::ValueHolder::CreateVoidGuarder("FreeMemory", outputs[i], {});
    outputs[i]->SetPlacement(faker->GetOutPlacement());
  }

  return {HyperStatus::Success(),
          {outputs[0]},                                      // ordered_holder
          {outputs.begin(), outputs.begin() + output_size},  // shape
          {outputs.begin() + output_size, outputs.end()}};   // addr
}

ge::graphStatus FakeExecuteNodeOutputsCreator(const ge::FastNode *node, KernelContext *context) {
  auto node_type = context->GetInputValue<const char *>(0);
  auto faker = NodesFakerForExe::GetNodeExeFaker(node_type);
  GE_ASSERT_NOTNULL(faker);
  faker->OutputCreator(node, context);
  return ge::GRAPH_SUCCESS;
}
ge::graphStatus FakeExecuteNode(KernelContext *context) {
  auto node_type = context->GetInputValue<const char *>(0);
  GE_ASSERT_NOTNULL(node_type);
  std::cout << "Execute faked kernel for node " << node_type << std::endl;
  auto faker = NodesFakerForExe::GetNodeExeFaker(node_type);
  GE_ASSERT_NOTNULL(faker);
  return faker->RunFunc(context);
}
}  // namespace
void NodesFakerForExe::FakeNode(const string &node_type, GertRuntimeStub *stub, int32_t placement) {
  stub->GetConverterStub().Register(node_type, FakedConverter, placement);
  stub->GetKernelStub().SetUp("FakeExecuteNode", {FakeExecuteNode, FakeExecuteNodeOutputsCreator, nullptr, nullptr});
}

std::list<std::pair<std::string, std::shared_ptr<BaseNodeExeFaker>>> ctx_fakers;

void NodesFakerForExe::PushCtxFaker(const std::string &node_type, std::shared_ptr<BaseNodeExeFaker> faker) {
  ctx_fakers.push_front(std::make_pair(node_type, faker));
}

void NodesFakerForExe::PopCtxFaker(const std::string &node_type) {
  for (auto &ctx_faker : ctx_fakers) {
    if (ctx_faker.first == node_type) {
      ctx_fakers.remove(ctx_faker);
      return;
    }
  }
}

BaseNodeExeFaker *NodesFakerForExe::GetNodeExeFaker(const string &node_type) {
  static std::map<std::string, std::shared_ptr<BaseNodeExeFaker>> types_to_faker {
      {"Shape", std::shared_ptr<BaseNodeExeFaker>(new FakedShapeImpl())},
      {"Rank", std::shared_ptr<BaseNodeExeFaker>(new FakedRankImpl())},
      {"Size", std::shared_ptr<BaseNodeExeFaker>(new FakedSizeImpl())},
  };
  for (auto &ctx_faker : ctx_fakers) {
    if (ctx_faker.first == node_type) {
      return ctx_faker.second.get();
    }
  }
  auto iter = types_to_faker.find(node_type);
  if (iter == types_to_faker.end()) {
    return nullptr;
  }
  return iter->second.get();
}
}  // namespace gert
