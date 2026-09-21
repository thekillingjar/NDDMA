/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph_builder/bg_reusable_stream_allocator.h"

#include <gtest/gtest.h>

#include "common/bg_test.h"
#include "common/summary_checker.h"
#include "graph/compute_graph.h"
#include "exe_graph/lowering/frame_selector.h"

namespace gert {
namespace bg {
class BgReusableStreamAllocatorUT : public BgTestAutoCreateFrame {
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

TEST_F(BgReusableStreamAllocatorUT, ReusableStreamAllocator_ok) {
  InitTestFrames();
  LoweringGlobalData global_data;
  const auto &allocator = ReusableStreamAllocator(global_data);
  ASSERT_NE(allocator, nullptr);
  const auto &allocator_in_init = HolderOnInit(allocator);
  ASSERT_NE(allocator_in_init, nullptr);
  ASSERT_EQ(allocator_in_init->GetFastNode()->GetDataOutNum(), 1U);

  // check ReusableStreamAllocator node on init graph
  const auto &init_exe_graph = init_frame->GetExecuteGraph().get();
  ExeGraphSummaryChecker(init_exe_graph)
      .StrictDirectNodeTypes(std::map<std::string, size_t>{{"ReusableStreamAllocator", 1}, {"InnerNetOutput", 1}});
}
}  // namespace bg
}  // namespace gert