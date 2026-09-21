/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "asc_backend_scheduler_adapter.h"
#include "common/checker.h"

#include "adaption_optimize_transpose_count.h"
#include "adaption_complete_node_attrs.h"
#include "adaption_fallback_load.h"
#include "torch_adaption_fallback_load.h"
#include "adaption_improve_precision.h"
#include "adaption_complete_asc_io_index.h"
#include "serialize_asc_backend.h"
#include "adaption_fallback_scalar.h"
#include "adaption_transpose_backward.h"
#include "adaption_optimized_fallback.h"
#include "adaption_combine_split.h"

namespace ge {
Status AscBackendSchedulerAdapter::DoBeforePass(const ComputeGraphPtr &graph) const {
  // 优化反推出的transpose数量能力
  GE_ASSERT_SUCCESS(asc_adapt::OptimizeTransposeCount(graph));

  // 由于后端schedule的限制，需要把ascgraph的node的输入输出tensor上loop轴补齐成和graph loop轴一样
  GE_ASSERT_SUCCESS(asc_adapt::CompleteNodeAttrsOnAscGraphForSched(graph));

  // 为codegen做transpose后移，把transpose数量变为1，codegen暂时只支持一个ascgraph只有一个transpose,反推暂时不插transpose
  // 需要在broadcast反推前做，load直连store，store有transpose场景，graph轴和load一致时先插broadcast在插transpose轴序有问题
  GE_ASSERT_SUCCESS(asc_adapt::TransposeBackwardForCodegen(graph));

  // 由于后端schedule的限制，需要还原broadcast view
  GE_ASSERT_SUCCESS(asc_adapt::OptimizedFallback(graph));

  // 把fp16和bf16的node改为fp32，以提高精度
  GE_ASSERT_SUCCESS(PrecisionImprover::ImprovePrecisionToFp32(graph));

  // 为scalar增加broadcast
  // GE_ASSERT_SUCCESS(asc_adapt::FallbackScalarToBroadcastWithoutCheckType(graph)); 统一在OptimizedFallback中插入broadcast

  return SUCCESS;
}

Status AscBackendSchedulerAdapter::DoAfterPass(const ComputeGraphPtr &graph) const {
  // 后端需要前端将split合并为一个split
  GE_ASSERT_SUCCESS(asc_adapt::SplitCombineForBackend(graph));

  // 给ascGraph的节点补充index属性
  GE_ASSERT_SUCCESS(asc_adapt::CompleteIndexAttrForAscGraph(graph));

  // 给ascGraph的节点序列化,torch路径不需要序列化
  if (asc_adapt::IsGeType()) {
    GE_ASSERT_SUCCESS(asc_adapt::SerilizeAscBackendNode(graph));
  }
  return SUCCESS;
}
}  // namespace ge
