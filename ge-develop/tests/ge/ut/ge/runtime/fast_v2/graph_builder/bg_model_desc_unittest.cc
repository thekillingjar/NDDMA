/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph_builder/bg_model_desc.h"
#include <gtest/gtest.h>
#include "ge_graph_dsl/graph_dsl.h"
#include "register/node_converter_registry.h"
#include "exe_graph_comparer.h"
#include "faker/global_data_faker.h"
#include "common/bg_test.h"
#include "register/op_impl_registry.h"
#include "common/share_graph.h"
#include "common/summary_checker.h"
#include "common/const_data_helper.h"

namespace gert {
namespace bg {
class BgModelDescUT : public BgTestAutoCreateFrame {
 public:
  GraphFrame *root_frame = nullptr;
  std::unique_ptr<GraphFrame> init_frame;
  std::unique_ptr<GraphFrame> de_init_frame;
  void InitTestFrames() {
    root_frame = ValueHolder::GetCurrentFrame();
    auto init_node = ValueHolder::CreateVoid<bg::ValueHolder>("Init", {});
    ValueHolder::PushGraphFrame(init_node, "Init");
    init_frame = ValueHolder::PopGraphFrame();

    auto de_init_node = ValueHolder::CreateVoid<bg::ValueHolder>("DeInit", {});
    ValueHolder::PushGraphFrame(de_init_node, "DeInit");
    de_init_frame = ValueHolder::PopGraphFrame();

    auto main_node = ValueHolder::CreateVoid<bg::ValueHolder>(GetExecuteGraphTypeStr(ExecuteGraphType::kMain), {});
    ValueHolder::PushGraphFrame(main_node, "Main");
  }
};

/*
*                  init_graph :                          main_graph:
*  ConstData(session)     ConstData(model_desc)          InnerData
*                                 |                          |
*                          GetSpaceRegistry              consumer1
*                                 |
*                           InnerNetOutput
*/
TEST_F(BgModelDescUT, GetSpaceRegistryOk) {
  InitTestFrames();
  LoweringGlobalData global_data;
  LowerConstDataNode(global_data);
  auto space_registry_holder = bg::GetSpaceRegistry(global_data);
  ASSERT_NE(space_registry_holder, nullptr);
  auto consumer2 = ValueHolder::CreateSingleDataOutput("consumer1", {space_registry_holder});

  auto main_frame = ValueHolder::PopGraphFrame();
  ASSERT_NE(init_frame, nullptr);

  auto init_exe_graph = init_frame->GetExecuteGraph().get();
  ExeGraphSummaryChecker(init_exe_graph).StrictDirectNodeTypes(
    std::map<std::string, size_t>{{"ConstData", 2}, {"GetSpaceRegistry", 1}, {"InnerNetOutput", 1}});

  auto main_exe_graph = main_frame->GetExecuteGraph().get();
  ExeGraphSummaryChecker(main_exe_graph).StrictDirectNodeTypes(std::map<std::string, size_t>{{"InnerData", 1}, {"consumer1", 1}});
}

}  // namespace bg
}  // namespace gert