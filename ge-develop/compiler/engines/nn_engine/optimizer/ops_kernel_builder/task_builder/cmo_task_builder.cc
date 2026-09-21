/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ops_kernel_builder/task_builder/cmo_task_builder.h"

namespace fe {
CMOTaskBuilder::CMOTaskBuilder() {}
CMOTaskBuilder::~CMOTaskBuilder() {}

Status CMOTaskBuilder::GenerateCMOTask(const ge::Node &node, std::vector<domi::TaskDef> &task_defs,
                                       TaskBuilderContext &context, const bool &pre_cmo_task) {
  ge::OpDescPtr op_desc = node.GetOpDesc();
  CmoExtraAttr cmo_ext_attr;
  cmo_ext_attr = op_desc->TryGetExtAttr(kOpExtattrNameCmo, cmo_ext_attr);

  Status status;
  for (auto &cmo_attr : cmo_ext_attr) {
    GenerateCMOTaskBasePtr cmo_task_ptr = GetCMOTaskType(node, context, cmo_attr.first, pre_cmo_task);
    if (cmo_task_ptr == nullptr) {
      FE_LOGD("No necessary to generate [%s] cmo task %s node[%s].",
              cmo_attr.first.c_str(), GetPosition(pre_cmo_task).c_str(), node.GetName().c_str());
      continue;
    }
    status = cmo_task_ptr->GenerateTask(task_defs, context.stream_id, cmo_attr.second);
    if (status != SUCCESS) {
      FE_LOGW("Generate [%s] CMO task for node[%s] failed.", cmo_attr.first.c_str(), node.GetName().c_str());
      return FAILED;
    }
  }
  return SUCCESS;
}

GenerateCMOTaskBasePtr CMOTaskBuilder::GetCMOTaskType(const ge::Node &node, TaskBuilderContext &context,
                                                      const std::string &task_type, const bool &pre_cmo_task) {
  if (pre_cmo_task) {
    if (task_type == kCmoPrefetch) {
      FE_MAKE_SHARED(generate_cmo_prefetch_task_ptr_ = std::make_shared<GenerateCMOPrefetchTask>(node, context),
                     return nullptr);
      return generate_cmo_prefetch_task_ptr_;
    } else if (task_type == kCmoBarrier) {
      FE_MAKE_SHARED(generate_cmo_barrier_task_ptr_ = std::make_shared<GenerateCMOBarrierTask>(node, context),
                     return nullptr);
      return generate_cmo_barrier_task_ptr_;
    }
  } else {
    if (task_type == kCmoInvalid) {
      FE_MAKE_SHARED(generate_cmo_invalid_task_ptr_ = std::make_shared<GenerateCMOInvalidTask>(node, context),
                     return nullptr);
      return generate_cmo_invalid_task_ptr_;
    }  else if (task_type == kCmoWriteback) {
      FE_MAKE_SHARED(generate_cmo_writeback_task_ptr_ = std::make_shared<GenerateCMOWritebackTask>(node, context),
                     return nullptr);
      return generate_cmo_writeback_task_ptr_;
    }
  }
  return nullptr;
}
std::string CMOTaskBuilder::GetPosition(const bool &pre_task) const {
  if (pre_task) {
    return "before";
  } else {
    return "after";
  }
}
}
