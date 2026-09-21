/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include "graph_builder/assign_device_mem.h"
#include "graph/utils/tensor_utils.h"
#include "graph/ge_attr_value.h"
#include "faker/global_data_faker.h"
#include "common/share_graph.h"
#include "exe_graph/lowering/value_holder.h"
#include "exe_graph/lowering/graph_frame.h"
#include "graph/utils/execute_graph_utils.h"

namespace gert {
class AssignWeightMemUT : public testing::Test {
 protected:
  void SetUp() override {
    InitFrame();
  }
  void TearDown() override {
    while (bg::ValueHolder::PopGraphFrame() != nullptr) {}
  }
  void InitFrame() {
    bg::ValueHolder::PushGraphFrame();
    root_frame = bg::ValueHolder::GetCurrentFrame();
    auto init_node = bg::ValueHolder::CreateVoid<bg::ValueHolder>("Init", {});
    bg::ValueHolder::PushGraphFrame(init_node, "Init");
    init_frame = bg::ValueHolder::PopGraphFrame();

    auto de_init_node = bg::ValueHolder::CreateVoid<bg::ValueHolder>("DeInit", {});
    bg::ValueHolder::PushGraphFrame(de_init_node, "DeInit");
    de_init_frame = bg::ValueHolder::PopGraphFrame();

    auto main_node = bg::ValueHolder::CreateVoid<bg::ValueHolder>(GetExecuteGraphTypeStr(ExecuteGraphType::kMain), {});
    bg::ValueHolder::PushGraphFrame(main_node, "Main");
  }
  bg::GraphFrame *root_frame;
  std::unique_ptr<bg::GraphFrame> init_frame;
  std::unique_ptr<bg::GraphFrame> de_init_frame;
};

TEST_F(AssignWeightMemUT, GetOrCreateMemAssignerSuccess) {
  auto graph = ShareGraph::BuildShapeToReshapeGraph();
auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).Build();
  auto const_node = graph->FindFirstNodeMatchType("Const");
  ASSERT_NE(const_node, nullptr);
  ge::ConstGeTensorPtr tensor;
  ge::AttrUtils::GetTensor(const_node->GetOpDesc(), "value", tensor);
  ASSERT_NE(tensor, nullptr);
  OuterWeightMem outer_weight_mem = {nullptr, 0U};
  auto outer_mem_holder = bg::FrameSelector::OnInitRoot([&]() -> std::vector<bg::ValueHolderPtr> {
    return {bg::ValueHolder::CreateConst(&outer_weight_mem, sizeof(outer_weight_mem))};
  });
  global_data.SetUniqueValueHolder("OuterWeightMem", outer_mem_holder[0]);
  auto holder = AssignDeviceMem::GetOrCreateMemAssigner(global_data);
  ASSERT_NE(holder, nullptr);
  auto create_allocator_node =
    ge::ExecuteGraphUtils::FindFirstNodeMatchType(init_frame->GetExecuteGraph().get(), "CreateMemAssigner");
  ASSERT_NE(create_allocator_node, nullptr);
}
} // namespace gert