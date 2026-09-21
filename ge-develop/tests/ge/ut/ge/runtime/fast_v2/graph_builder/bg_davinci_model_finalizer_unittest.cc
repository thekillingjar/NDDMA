/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph_builder/bg_davinci_model_finalizer.h"

#include <gtest/gtest.h>

#include "common/bg_test.h"
#include "common/summary_checker.h"
#include "exe_graph/lowering/frame_selector.h"
#include "graph/compute_graph.h"

namespace gert {
namespace bg {
class BgDavinciModelFinalizerUT : public BgTestAutoCreateFrame {
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

TEST_F(BgDavinciModelFinalizerUT, FinalizeDavinciModel_ok) {
  InitTestFrames();
  const auto &davinci_model_create = FrameSelector::OnInitRoot([]() -> std::vector<ValueHolderPtr> {
    return {ValueHolder::CreateSingleDataOutput("DavinciModelCreate", {}),
            ValueHolder::CreateSingleDataOutput("DavinciModelCreate", {})};
  });
  EXPECT_EQ(davinci_model_create.size(), 2);

  LoweringGlobalData global_data;
  std::vector<ValueHolderPtr> davinci_model_finalizer_list;
  for (size_t i = 0U; i < 2U; ++i) {
    davinci_model_finalizer_list.emplace_back(DavinciModelFinalizer(davinci_model_create[i], global_data));
    ASSERT_NE(davinci_model_finalizer_list.back(), nullptr);
    ASSERT_NE(davinci_model_finalizer_list.back()->GetFastNode(), nullptr);
    ASSERT_EQ(davinci_model_finalizer_list.back()->GetFastNode()->GetDataInNum(), i + 1U);
  }
  ASSERT_EQ(davinci_model_finalizer_list[0].get(), davinci_model_finalizer_list[1].get());

  // check DavinciModelCreate node on init graph
  const auto &init_exe_graph = init_frame->GetExecuteGraph().get();
  ExeGraphSummaryChecker(init_exe_graph)
      .StrictDirectNodeTypes(std::map<std::string, size_t>{{"DavinciModelCreate", 2}, {"InnerNetOutput", 1}});

  // check DavinciModelFinalizer node on de_init graph
  const auto &de_init_exe_graph = de_init_frame->GetExecuteGraph().get();
  ExeGraphSummaryChecker(de_init_exe_graph)
      .StrictDirectNodeTypes(std::map<std::string, size_t>{{"InnerData", 2}, {"DavinciModelFinalizer", 1}});
}
}  // namespace bg
}  // namespace gert