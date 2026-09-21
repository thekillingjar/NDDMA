/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "bg_davinci_model_finalizer.h"

#include "exe_graph/lowering/frame_selector.h"
#include "exe_graph/lowering/value_holder_utils.h"

namespace gert {
namespace bg {
namespace {
const std::string kDavinciModelFinalizer = "DavinciModelFinalizer";
bool IsFreeFixedFeatureMemory(const ge::FastNode *const node) {
  return (strcmp(node->GetTypePtr(), "FreeFixedFeatureMemory") == 0);
}
}
ValueHolderPtr DavinciModelFinalizer(const ValueHolderPtr &davinci_model_holder, LoweringGlobalData &global_data) {
  auto builder = []() -> std::vector<bg::ValueHolderPtr> {
    return FrameSelector::OnDeInitRoot([]() -> std::vector<ValueHolderPtr> {
      return {ValueHolder::CreateVoid<ValueHolder>(kDavinciModelFinalizer.c_str(), {})};
    });
  };
  const auto &davinci_model_finalizer_list = global_data.GetOrCreateUniqueValueHolder(kDavinciModelFinalizer, builder);
  GE_ASSERT_TRUE(!davinci_model_finalizer_list.empty());
  const auto &davinci_model_finalizer = davinci_model_finalizer_list[0];
  GE_ASSERT_NOTNULL(davinci_model_finalizer);

  GE_ASSERT_NOTNULL(davinci_model_holder);
  GE_ASSERT_TRUE(ValueHolderUtils::IsNodeValid(davinci_model_holder));
  GE_ASSERT_SUCCESS(davinci_model_finalizer->AppendInputs({davinci_model_holder}));
  GELOGD("Success to add data edge from davinci_model %s to davinci_model_finalizer.",
         ValueHolderUtils::GetNodeNameBarePtr(davinci_model_holder));

  const auto exe_graph = davinci_model_finalizer->GetExecuteGraph();
  GE_ASSERT_NOTNULL(exe_graph);
  const auto free_fixed_fm_nodes = exe_graph->GetAllNodes(IsFreeFixedFeatureMemory);
  for (const auto free_node : free_fixed_fm_nodes) {
    GE_ASSERT_NOTNULL(exe_graph->AddEdge(davinci_model_finalizer->GetFastNode(), ge::kControlEdgeIndex,
                                         free_node, ge::kControlEdgeIndex));
    GELOGI("success to add ctrl edge from davinci_model_finalizer to free node: %s", free_node->GetNamePtr());
  }
  return davinci_model_finalizer;
}
}  // namespace bg
}  // namespace gert
