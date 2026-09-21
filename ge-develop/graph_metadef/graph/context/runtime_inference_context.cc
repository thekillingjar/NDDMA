/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <utility>

#include "graph/utils/tensor_adapter.h"
#include "framework/common/debug/ge_log.h"
#include "graph/runtime_inference_context.h"

namespace ge {
void RuntimeInferenceContext::Release() {
  const std::lock_guard<std::mutex> lk(mu_);
  ge_tensors_.clear();
}

graphStatus RuntimeInferenceContext::SetTensor(int64_t node_id, int32_t output_id, GeTensorPtr tensor) {
  const std::lock_guard<std::mutex> lk(mu_);
  auto &output_ge_tensors = ge_tensors_[node_id];
  if (static_cast<size_t>(output_id) >= output_ge_tensors.size()) {
    const size_t output_tensor_size = static_cast<size_t>(output_id) + 1U;
    output_ge_tensors.resize(output_tensor_size);
  }

  GELOGD("Set tensor for node_id = %" PRId64 ", output_id = %" PRId32, node_id, output_id);
  output_ge_tensors[static_cast<size_t>(output_id)] = std::move(tensor);

  return GRAPH_SUCCESS;
}

graphStatus RuntimeInferenceContext::GetTensor(const int64_t node_id, int32_t output_id, GeTensorPtr &tensor) const {
  if (output_id < 0) {
    REPORT_INNER_ERR_MSG("E18888", "Invalid output index: %d", output_id);
    GELOGE(GRAPH_PARAM_INVALID, "[Check][Param] Invalid output index: %d", output_id);
    return GRAPH_PARAM_INVALID;
  }

  const std::lock_guard<std::mutex> lk(mu_);
  const auto iter = ge_tensors_.find(node_id);
  if (iter == ge_tensors_.end()) {
    GELOGW("Node not register. Id = %" PRId64, node_id);
    return INTERNAL_ERROR;
  }

  auto &output_tensors = iter->second;
  if (static_cast<uint32_t>(output_id) >= output_tensors.size()) {
    GELOGW("The %" PRId32 " th output tensor for node id [%" PRId64 "] has not been registered.", output_id, node_id);
    return GRAPH_FAILED;
  }

  GELOGD("Get ge tensor for node_id = %" PRId64 ", output_id = %" PRId32, node_id, output_id);
  tensor = output_tensors[static_cast<size_t>(output_id)];
  if (tensor == nullptr) {
    GELOGW("The %" PRId32 " th output tensor registered for node id [%" PRId64 "] is nullptr.", output_id, node_id);
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}
} // namespace ge
