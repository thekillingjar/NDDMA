/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "data_flow_executor_utils.h"
#include "framework/common/debug/ge_log.h"
#include "graph_metadef/common/ge_common/util.h"
#include "graph/utils/tensor_utils.h"

namespace ge {
Status DataFlowExecutorUtils::FillRuntimeTensorDesc(const GeTensorDesc &tensor_desc,
                                                    RuntimeTensorDesc &runtime_tensor_desc, bool calc_size) {
  const auto &shape = tensor_desc.GetShape();
  // if original shape is not set, it's a scalar
  const auto &original_shape = tensor_desc.IsOriginShapeInitialized() ? tensor_desc.GetOriginShape() : shape;
  GELOGD("type = %d, shape = %s, has_origin_shape = %d, original shape = %s",
         static_cast<int32_t>(tensor_desc.GetDataType()),
         ToString(shape.GetDims()).c_str(),
         static_cast<int32_t>(tensor_desc.IsOriginShapeInitialized()),
         ToString(original_shape.GetDims()).c_str());
  size_t num_dims = shape.GetDimNum();
  GE_CHECK_LE(num_dims, static_cast<size_t>(kMaxDimSize));
  runtime_tensor_desc.shape[0U] = static_cast<int64_t>(num_dims);
  for (size_t i = 0U; i < shape.GetDimNum(); ++i) {
    runtime_tensor_desc.shape[i + 1U] = shape.GetDim(i);
  }

  num_dims = original_shape.GetDimNum();
  GE_CHECK_LE(num_dims, static_cast<size_t>(kMaxDimSize));
  runtime_tensor_desc.original_shape[0U] = static_cast<int64_t>(num_dims);
  for (size_t i = 0U; i < original_shape.GetDimNum(); ++i) {
    runtime_tensor_desc.original_shape[i + 1U] = original_shape.GetDim(i);
  }
  runtime_tensor_desc.dtype = static_cast<int64_t>(tensor_desc.GetDataType());
  if (calc_size) {
    int64_t tensor_raw_size = -1;
    GE_CHK_GRAPH_STATUS_RET(TensorUtils::CalcTensorMemSize(tensor_desc.GetShape(),
                                                           tensor_desc.GetFormat(),
                                                           tensor_desc.GetDataType(),
                                                           tensor_raw_size),
                            "Failed to calc tensor raw size");
    runtime_tensor_desc.data_size = static_cast<uint64_t>(tensor_raw_size);
  }
  return SUCCESS;
}
}
