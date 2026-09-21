/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dflow/inc/data_flow/model/graph_model.h"
#include "framework/common/helper/model_helper.h"

namespace ge {

GraphModel::~GraphModel() {
  delete[] static_cast<char *>(model_data_.model_data);
  model_data_.model_data = nullptr;
}

GraphModel::GraphModel(const ComputeGraphPtr compute_graph)
    : PneModel(compute_graph) {
  PneModel::SetModelName(compute_graph->GetName());
  PneModel::SetModelId(compute_graph->GetGraphID());
}

Status GraphModel::Init(const ModelData &model_data) {
  char *const model_data_addr = new (std::nothrow) char[model_data.model_len];
  if (model_data_addr == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "new an object failed.");
    GELOGE(FAILED, "new an object failed. Length: [%zu].", model_data.model_len);
    return FAILED;
  }
  if (memcpy_s(model_data_addr, model_data.model_len, model_data.model_data, model_data.model_len) != EOK) {
      GELOGE(FAILED, "Failed to copy data, dst size=%zu, src size=%zu", model_data.model_len, model_data.model_len);
      delete[] static_cast<char *>(model_data_addr);
      model_data_.model_data = nullptr;
      return FAILED;
  }
  model_data_.model_data = static_cast<void*>(model_data_addr);
  model_data_.model_len = model_data.model_len;
  return SUCCESS;
}

Status GraphModel::SerializeModel(ModelBufferData &model_buffer) {
  model_buffer.length = model_data_.model_len;
  model_buffer.data = std::shared_ptr<uint8_t>(PtrToPtr<void, uint8_t>(model_data_.model_data),
                                                   [](const uint8_t *const pointer) { (void)pointer; });
  GELOGD("[Serialize][Submodel] succeeded, model_name = [%s], size = %lu", 
                  GetModelName().c_str(), model_buffer.length);
  return SUCCESS;
}

Status GraphModel::UnSerializeModel(const ModelBufferData &model_buffer) {
  (void)model_buffer;
  return SUCCESS;
}
}  // namespace ge