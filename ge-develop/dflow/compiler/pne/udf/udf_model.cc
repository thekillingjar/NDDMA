/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "udf_model.h"

#include "common/plugin/ge_make_unique_util.h"
#include "framework/common/debug/ge_log.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/debug/ge_attr_define.h"
#include "graph_metadef/graph/utils/file_utils.h"
#include "graph_metadef/common/ge_common/util.h"

namespace ge {
UdfModel::UdfModel(const ComputeGraphPtr &root_graph) : PneModel(root_graph){};

Status UdfModel::SerializeModel(ModelBufferData &model_buff) {
  if (GetIsBuiltinModel()) {
    return SerializeModelDef(model_buff);
    } else {
      uint32_t data_len = 0U;
      std::string saved_model_path = GetSavedModelPath();
      auto buff = GetBinFromFile(saved_model_path, data_len);
      GE_CHECK_NOTNULL(buff, ". Open file fail, path=%s", saved_model_path.c_str());
      model_buff.data.reset(reinterpret_cast<uint8_t *>(buff.release()), std::default_delete<uint8_t[]>());
      model_buff.length = static_cast<uint64_t>(data_len);
  }
  return SUCCESS;
}

Status UdfModel::SerializeModelDef(ModelBufferData &model_buff) const{
  const size_t size = udf_model_def_.ByteSizeLong();
  GELOGI("Serialize model def size is %zu.", size);
  GE_CHK_BOOL_RET_STATUS(size != 0, FAILED, "Model is empty.");
  auto buff = MakeUnique<uint8_t[]>(size);
  if (buff == nullptr) {
    GELOGE(FAILED, "Alloc %zu memory for UdfModelDef failed.", size);
    return FAILED;
  }
  if (!udf_model_def_.SerializeToArray(buff.get(), size)) {
    GELOGE(FAILED, "Failed to serialize UdfModelDef To array.");
    return FAILED;
  }
  model_buff.data.reset(buff.release(), std::default_delete<uint8_t[]>());
  model_buff.length = static_cast<uint64_t>(size);
  return SUCCESS;
}

Status UdfModel::UnSerializeModel(const ModelBufferData &model_buff) {
  if (!GetIsBuiltinModel()) {
    GELOGE(FAILED, "User define udf does not support unserialize model.");
    return FAILED;
  }
  GE_CHECK_NOTNULL(model_buff.data);
  GE_CHK_BOOL_RET_STATUS(udf_model_def_.ParseFromArray(model_buff.data.get(), model_buff.length), FAILED,
                         "Failed to parse UdfModelDef from array.");
  return SUCCESS;
}

std::string UdfModel::GetLogicDeviceId() const {
  return GetLogicDeviceId(false);
}

std::string UdfModel::GetRedundantLogicDeviceId() const {
  return GetLogicDeviceId(true);
}

std::string UdfModel::GetLogicDeviceId(bool is_redundant) const {
  const auto &graph = GetRootGraph();
  if (graph == nullptr) {
    GELOGD("udf model graph is null, return empty logic device id.");
    return "";
  }
  std::string logic_device_id;
  const auto attr_name = is_redundant ? ATTR_NAME_REDUNDANT_LOGIC_DEV_ID : ATTR_NAME_LOGIC_DEV_ID;
  if ((!AttrUtils::GetStr(graph, attr_name, logic_device_id))) {
    GELOGD("graph[%s] has not set logic device id, redundant flag[%d].",
           graph->GetName().c_str(), static_cast<int32_t>(is_redundant));
    return "";
  }
  return logic_device_id;
}
}  // namespace ge
