/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <bitset>
#include <iostream>
#include "graph/types.h"
#include "graph/def_types.h"
#include "axis_constants.h"
#include "exe_graph/runtime/tiling_context.h"
#include "exe_graph/runtime/extended_kernel_context.h"
#include "exe_graph/runtime/shape.h"
#include "exe_graph/lowering/getcdim.h"

namespace gert {
  const std::map<ge::Format, const int32_t> CDIM_INDEX_OF_FORMAT {
    {ge::FORMAT_NCHW, transformer::AXIS_NCHW_DIM_C},
    {ge::FORMAT_HWCN, transformer::AXIS_HWCN_DIM_C},
    {ge::FORMAT_NHWC, transformer::AXIS_NHWC_DIM_C},
    {ge::FORMAT_CHWN, transformer::AXIS_CHWN_DIM_C},
    {ge::FORMAT_NDHWC, transformer::NDHWC_DIM_C},
    {ge::FORMAT_NCDHW, transformer::NCDHW_DIM_C},
    {ge::FORMAT_DHWCN, transformer::DHWCN_DIM_C},
    {ge::FORMAT_DHWNC, transformer::DHWNC_DIM_C}
  };

namespace {
  int64_t GetCDim(TilingContext *const context, const size_t index, const bool is_input) {
    if (context == nullptr) {
      return -1;
    }
    auto extend_context = ge::PtrToPtr<TilingContext, ExtendedKernelContext>(context);
    auto compute_node_info = extend_context->GetComputeNodeInfo();
    if (compute_node_info == nullptr) {
      return -1;
    }
    auto kernel_context = ge::PtrToPtr<TilingContext, KernelContext>(context);
    const CompileTimeTensorDesc *td = nullptr;
    StorageShape *storage_shape = nullptr;
    if (is_input) {
      td = compute_node_info->GetInputTdInfo(index);
      storage_shape = kernel_context->MutableInputPointer<StorageShape>(index);
    } else {
      td = compute_node_info->GetOutputTdInfo(index);
      storage_shape = kernel_context->GetOutputPointer<StorageShape>(index);
    }
    if ((td == nullptr) || (storage_shape == nullptr)) {
      return -1;
    }
    const auto original_format = td->GetOriginFormat();
    const auto iter = CDIM_INDEX_OF_FORMAT.find(original_format);
    if (iter == CDIM_INDEX_OF_FORMAT.cend()) {
      return -1;
    }
    Shape &origin_shape = storage_shape->MutableOriginShape();
    const auto expend_dims = td->GetExpandDimsType();
    Shape expand_shape;
    (void) expend_dims.Expand(origin_shape, expand_shape);

    if (static_cast<size_t>(iter->second) >= expand_shape.GetDimNum()) {
      return -1;
    }
    if (expand_shape.GetDimNum() == origin_shape.GetDimNum()) {
      return static_cast<int64_t>(origin_shape.GetDim(iter->second));
    } else {
      return static_cast<int64_t>(expand_shape.GetDim(iter->second));
    }
  }
}  // namespace

  int64_t GetInputCDim(TilingContext *kernel_context, const size_t index) {
    return GetCDim(kernel_context, index, true);
  }
  int64_t GetOutputCDim(TilingContext *kernel_context, const size_t index) {
    return GetCDim(kernel_context, index, false);
  }
}  // namespace gert
