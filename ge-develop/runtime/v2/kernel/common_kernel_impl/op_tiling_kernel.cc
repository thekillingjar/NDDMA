/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "op_tiling_kernel.h"
#include "graph/ge_error_codes.h"
#include "graph/node.h"
#include "register/kernel_registry.h"
#include "exe_graph/runtime/continuous_vector.h"
#include "common/checker.h"
#include "tiling.h"
#include "exe_graph/runtime/gert_tensor_data.h"
#include "core/executor/multi_thread_topological/executor/schedule/producer/producers/kernel_tags/critical_section_config.h"
#include "exe_graph/runtime/gert_mem_allocator.h"

namespace gert {
namespace kernel {
ge::graphStatus CreateTensorDataOutputs(const ge::FastNode *node, KernelContext *context) {
  for (uint32_t i = 0U; i < kOpTilingOutputSize; ++i) {
    auto av = context->GetOutput(i);
    GE_ASSERT_NOTNULL(av);
    auto tensor_data = new (std::nothrow) GertTensorData(0U, kOnHost, -1, nullptr);
    if (tensor_data == nullptr) {
      GELOGE(ge::FAILED, "Failed create addr outputs at index:%u for node %s", i, node->GetName().c_str());
      return ge::GRAPH_FAILED;
    }
    av->SetWithDefaultDeleter(tensor_data);
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus CreateOpTilingShapeOutputs(const ge::FastNode *node, KernelContext *context) {
  for (uint32_t i = 0U; i < kOpTilingOutputSize; ++i) {
    auto av = context->GetOutput(i);
    GE_ASSERT_NOTNULL(av);
    auto storage_shape = new (std::nothrow) StorageShape();
    if (storage_shape == nullptr) {
      GELOGE(ge::FAILED, "Failed create shape outputs at index:%u for node %s", i, node->GetName().c_str());
      return ge::GRAPH_FAILED;
    }
    if (kTilingOutputIndexToInputIndex[i] != TilingContext::kOutputTilingData) {
      storage_shape->MutableStorageShape().SetDimNum(0UL);
      storage_shape->MutableOriginShape().SetDimNum(0UL);
    }
    av->SetWithDefaultDeleter(storage_shape);
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus BuildOpTilingTensorDataInner(const void *input_addr, const uint32_t index, KernelContext *context) {
  auto stream_id = context->GetInputPointer<int64_t>(static_cast<size_t>(OpTilingExtendInputs::kStreamId));
  GE_ASSERT_NOTNULL(stream_id);
  if (input_addr == nullptr) {
    GELOGE(ge::GRAPH_PARAM_INVALID,
           "Index:%u of input is a invalid nullptr of Optiling(0-data 1-key 2-blockdim 3-cond).", index);
    return ge::GRAPH_PARAM_INVALID;
  }
  auto out_tensor_data = context->GetOutputPointer<GertTensorData>(index);
  if (out_tensor_data == nullptr) {
    GELOGE(ge::GRAPH_PARAM_INVALID,
           "Index:%u of output is a invalid nullptr of Optiling(0-data 1-key 2-blockdim 3-cond).", index);
    return ge::GRAPH_PARAM_INVALID;
  }
  *out_tensor_data = GertTensorData{const_cast<void *>(input_addr), out_tensor_data->GetSize(),
                                    out_tensor_data->GetPlacement(), *stream_id};
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus BuildOpTilingUnmanagedTensorData(KernelContext *context) {
  // the node before is tiling fuction. Using tiling output to construct current input
  int32_t tiling_cond = context->GetInputValue<int32_t>(TilingContext::kOutputTilingCond);
  if (tiling_cond < 0) {
    GELOGE(ge::GRAPH_PARAM_INVALID, "kOutputTilingCond value：%d is invalid which should be >= 0.", tiling_cond);
    return ge::GRAPH_PARAM_INVALID;
  }
  for (uint32_t i = 0U; i < kOpTilingOutputSize; ++i) {
    // block dim type is uint32 and Tiling cond is int32, both of them are less than size of void*
    if ((kTilingOutputIndexToInputIndex[i] == TilingContext::kOutputTilingCond) ||
        (kTilingOutputIndexToInputIndex[i] == TilingContext::kOutputBlockDim)) {
      auto input_addr = context->GetInputPointer<void *>(kTilingOutputIndexToInputIndex[i]);
      GE_ASSERT_SUCCESS(BuildOpTilingTensorDataInner(input_addr, i, context));
    } else {
      auto input_addr = context->GetInputValue<void *>(kTilingOutputIndexToInputIndex[i]);
      GE_ASSERT_SUCCESS(BuildOpTilingTensorDataInner(input_addr, i, context));
    }
  }
  return ge::GRAPH_SUCCESS;
}

REGISTER_KERNEL(BuildOpTilingUnmanagedTensorData)
    .RunFunc(BuildOpTilingUnmanagedTensorData)
    .OutputsCreator(CreateTensorDataOutputs);

ge::graphStatus BuildOpTilingOutputShape(KernelContext *context) {
  // the node before is tiling fuction. Using tiling output to construct current input
  auto tiling_data = context->GetInputValue<TilingData *>(TilingContext::kOutputTilingData);
  if (tiling_data == nullptr) {
    GELOGE(ge::FAILED, "Get tilling data nullptr result for build output shape failed.");
    return ge::GRAPH_PARAM_INVALID;
  }
  const size_t data_size = tiling_data->GetDataSize();
  if (data_size > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
    GELOGE(ge::FAILED, "Data size:%zu is overflow to construct storage shape.", data_size);
    return ge::GRAPH_PARAM_INVALID;
  }
  auto storage_shape = context->GetOutputPointer<StorageShape>(kTilingDataIndexOfOpTiling);
  if (storage_shape == nullptr) {
    GELOGE(ge::GRAPH_PARAM_INVALID, "Shape of tiling data output is nullptr");
    return ge::GRAPH_PARAM_INVALID;
  }
  storage_shape->MutableStorageShape().SetDimNum(1UL);
  storage_shape->MutableOriginShape().SetDimNum(1UL);
  storage_shape->MutableStorageShape().SetDim(0UL, data_size);
  storage_shape->MutableOriginShape().SetDim(0UL, data_size);
  return ge::GRAPH_SUCCESS;
}

REGISTER_KERNEL(BuildOpTilingOutputShape)
    .RunFunc(BuildOpTilingOutputShape)
    .OutputsCreator(CreateOpTilingShapeOutputs);
}  // namespace kernel
}  // namespace gert
