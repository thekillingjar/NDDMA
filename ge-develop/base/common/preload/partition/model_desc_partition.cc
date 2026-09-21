/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph_metadef/common/ge_common/util.h"
#include "common/preload/partition/model_desc_partition.h"

namespace ge {
const std::string kAttrIsGraphPrefetch = "_is_graph_prefetch";
Status ModelDescPartition::GenModelDescInfo(const GeModelPtr &ge_model, ModelDescInfo &model_desc_info) const {
  const auto &model_task_def = ge_model->GetModelTaskDefPtr();
  const uint32_t task_num = static_cast<uint32_t>(model_task_def->task_size());
  model_desc_info.task_num = 0U;
  for (uint32_t task_index = 0U; task_index < task_num; task_index++) {
    const auto &task_def = model_task_def->task(static_cast<int32_t>(task_index));
    if (static_cast<ModelTaskType>(task_def.type()) != ModelTaskType::MODEL_TASK_KERNEL) {
      GELOGW("task has no kernel, skip it.task type is %u", static_cast<uint32_t>(task_def.type()));
      continue;
    }
    model_desc_info.task_num++;
  }
  model_desc_info.workspace_size = static_cast<uint64_t>(model_task_def->memory_size());
  model_desc_info.weight_size = static_cast<uint64_t>(model_task_def->weight_size());
  model_desc_info.weight_type = ge::WeightType::PREFETCH_EVERYTIME;
  ComputeGraphPtr compute_graph = ge_model->GetGraph();
  bool is_graph_prefetch = false;
  (void)ge::AttrUtils::GetBool(compute_graph, kAttrIsGraphPrefetch, is_graph_prefetch);
  if (is_graph_prefetch) {
    model_desc_info.weight_type = ge::WeightType::PREFETCH_ALL;
    GELOGI("Graph prefetch enable");
  }
  model_desc_info.profile_enable = false;
  model_desc_info.model_interrupt = false;

  GELOGD("task_num:%u, workspace_size:%llu, weight_size:%llu",
         model_desc_info.task_num, model_desc_info.workspace_size, model_desc_info.weight_size);
  return SUCCESS;
}

Status ModelDescPartition::SaveModelDescInfo(const ModelDescInfo model_desc_info) {
  GELOGD("[ModelDescPart]start save, des_addr_:%p, des_size_:%u", des_addr_, des_size_);

  const auto ret = memcpy_s(des_addr_, static_cast<uint64_t>(des_size_), &model_desc_info, sizeof(ModelDescInfo));
  GE_ASSERT_TRUE((ret == static_cast<int32_t>(ge::SUCCESS)), "memcpy_s failed, return: %d", ret);

  des_addr_ = PtrToPtr<void, uint8_t>(ValueToPtr(PtrToValue(des_addr_) + sizeof(ModelDescInfo)));
  des_size_ += static_cast<uint32_t>(sizeof(ModelDescInfo));
  GELOGD("[ModelDescPart]end save, des_addr_:%p, des_size_:%u", des_addr_, des_size_);

  return SUCCESS;
}

uint32_t ModelDescPartition::GetModelDescInfoSize() const {
  return static_cast<uint32_t>(sizeof(ModelDescInfo));
}

Status ModelDescPartition::Init(const GeModelPtr &ge_model, const uint8_t type) {
  (void)type;
  GELOGD("begin to generate model desc.");
  // get original info and buff_size
  ModelDescInfo model_desc_info;
  GE_ASSERT_SUCCESS(GenModelDescInfo(ge_model, model_desc_info));
  buff_size_ += GetModelDescInfoSize();

  // alloc buff
  buff_.reset(new (std::nothrow) uint8_t[buff_size_], std::default_delete<uint8_t[]>());
  GE_CHECK_NOTNULL(buff_);
  des_addr_ = buff_.get();
  des_size_ = buff_size_;

  // save buff
  GE_CHK_STATUS_RET(SaveModelDescInfo(model_desc_info), "[Save][ModelDescInfo] failed");
  GELOGD("end generate model desc, buff_size_:%u", buff_size_);
  return SUCCESS;
}
}  // namespace ge
