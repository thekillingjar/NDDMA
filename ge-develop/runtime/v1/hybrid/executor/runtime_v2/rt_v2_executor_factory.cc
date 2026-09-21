/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "rt_v2_executor_factory.h"

#include "framework/common/types.h"
#include "graph/debug/ge_attr_define.h"
#include "rt_v2_simple_executor.h"
#include "rt_v2_pipeline_executor.h"
#include "hybrid/model/hybrid_model.h"
#include "runtime/model_v2_executor.h"
#include "lowering/model_converter.h"

namespace {
ge::Status IsGraphStagePartitioned(const ge::ComputeGraphPtr &graph, bool &stage_partitioned) {
  for (auto &node : graph->GetDirectNode()) {
    GE_ASSERT_NOTNULL(node);
    GE_ASSERT_NOTNULL(node->GetOpDesc());
    if (node->GetType() == ge::PARTITIONEDCALL && ge::AttrUtils::HasAttr(node->GetOpDesc(), ge::ATTR_STAGE_LEVEL)) {
      stage_partitioned = true;
      return ge::SUCCESS;
    }
  }
  return ge::SUCCESS;
}
}  // namespace

namespace gert {
std::unique_ptr<RtV2ExecutorInterface> RtV2ExecutorFactory::Create(const ge::GeRootModelPtr &model,
                                                                   ge::DevResourceAllocator &allocator,
                                                                   RtSession *session) {
  GE_ASSERT_NOTNULL(model);
  GE_ASSERT_NOTNULL(model->GetRootGraph());
  bool stage_partitioned = false;
  GE_ASSERT_SUCCESS(IsGraphStagePartitioned(model->GetRootGraph(), stage_partitioned));
  if (!stage_partitioned) {
    return RtV2SimpleExecutor::Create(model, allocator, session);
  }
  return RtV2PipelineExecutor::Create(model, allocator, session);
}
}  // namespace gert
