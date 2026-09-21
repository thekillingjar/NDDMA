/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_TESTS_UT_GE_RUNTIME_V2_COMMON_CONST_DATA_H_
#define AIR_CXX_TESTS_UT_GE_RUNTIME_V2_COMMON_CONST_DATA_H_
#include "faker/global_data_faker.h"
#include "framework/runtime/gert_const_types.h"
#include "exe_graph/lowering/frame_selector.h"
namespace gert {
namespace bg {
inline void LowerConstDataNode(LoweringGlobalData &global_data) {
  size_t const_data_num = static_cast<size_t>(ConstDataType::kTypeEnd);
  auto const_data_outputs = bg::FrameSelector::OnInitRoot([&]() -> std::vector<bg::ValueHolderPtr> {
    std::vector<bg::ValueHolderPtr> const_datas;
    for (size_t i = 0U; i < const_data_num; ++i) {
      auto const_data_holder = bg::ValueHolder::CreateConstData(static_cast<int64_t>(i));    
      const_datas.emplace_back(const_data_holder);
    }
    return const_datas;
  });
  for (size_t i = 0U; i < const_data_num; ++i) {
    const auto &const_data_name = GetConstDataTypeStr(static_cast<ConstDataType>(i));
    global_data.SetUniqueValueHolder(const_data_name, const_data_outputs[i]);
  }
}

inline void CreateStreamNodeOnInitGraph(LoweringGlobalData &global_data) {
  auto stream_init_output = bg::FrameSelector::OnInitRoot([&]() -> std::vector<bg::ValueHolderPtr> {
    global_data.LoweringAndSplitRtStreams(1);
    return {bg::ValueHolder::CreateFeed(static_cast<int64_t>(ExecuteArgIndex::kStream))};
  });
}

inline void InitTestFrames() {
  auto init_node = ValueHolder::CreateVoid<bg::ValueHolder>("Init", {});
  ValueHolder::PushGraphFrame(init_node, "Init");
  ValueHolder::PopGraphFrame();

  auto de_init_node = ValueHolder::CreateVoid<bg::ValueHolder>("DeInit", {});
  ValueHolder::PushGraphFrame(de_init_node, "DeInit");
  ValueHolder::PopGraphFrame();

  auto main_node = ValueHolder::CreateVoid<bg::ValueHolder>(GetExecuteGraphTypeStr(ExecuteGraphType::kMain), {});
  ValueHolder::PushGraphFrame(main_node, "Main");
}
}
}  // namespace gert
#endif
