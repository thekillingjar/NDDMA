/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "exe_graph/runtime/extended_kernel_context.h"
#include "graph/ge_error_codes.h"
#include "register/kernel_registry_impl.h"
#include "kernel/tensor_attr.h"
#include "kernel/kernel_log.h"
#include "core/debug/kernel_tracing.h"

namespace gert {
namespace kernel {
ge::graphStatus CheckOutputShapesEmpty(KernelContext *context) {
  if (context == nullptr) {
    GELOGE(ge::FAILED, "[R2][Kernel][CheckOutShape]Context is null.");
    return ge::GRAPH_FAILED;
  }
  auto extend_context = reinterpret_cast<ExtendedKernelContext *>(context);
  auto compute_node_info = extend_context->GetComputeNodeInfo();
  if (compute_node_info == nullptr) {
    GELOGE(ge::FAILED, "[R2][Kernel][CheckOutShape]Compute node is null.");
    return ge::GRAPH_FAILED;
  }
  size_t out_num = compute_node_info->GetOutputsNum();
  // no output do as normal
  bool all_empty = (out_num != 0U);
  for (size_t i = 0U; i < out_num; ++i) {
    auto storage_shape = context->GetInputPointer<StorageShape>(i);
    if (storage_shape == nullptr) {
      GELOGE(ge::FAILED, "[R2][Kernel][CheckOutShape]Storage shape of index:%zu is null.", i);
      return ge::GRAPH_FAILED;
    }
    auto &shape = storage_shape->GetStorageShape();
    bool empty = false;
    for (size_t j = 0U, dim_num = shape.GetDimNum(); j < dim_num; ++j) {
      if (shape[j] == 0) {
        empty = true;
        break;
      }
    }
    if (!empty) {
      all_empty = false;
      break;
    }
  }
  auto cond = context->GetOutputPointer<uint32_t>(0);
  if (cond == nullptr) {
    GELOGE(ge::FAILED, "[R2][Kernel][CheckOutShape]Output pointer is null.");
    return ge::GRAPH_FAILED;
  }
  if (all_empty) {
    (*cond) = 0U;
    KERNEL_TRACE(
        "All outputs of Node[%s] are empty tensors. Setting ge::OPTION_ALL_TENSOR_NOT_EMPTY to 1 is not allowed.",
        compute_node_info->GetNodeName());
  } else {
    (*cond) = 1U;
    KERNEL_TRACE("Not all outputs of Node[%s] are empty tensors", compute_node_info->GetNodeName());
  }
  return ge::GRAPH_SUCCESS;
}
REGISTER_KERNEL(CheckOutputShapesEmpty).RunFunc(CheckOutputShapesEmpty);
}  // namespace kernel
}  // namespace gert
