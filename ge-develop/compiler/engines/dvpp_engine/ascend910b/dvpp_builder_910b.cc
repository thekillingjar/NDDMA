/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dvpp_builder_910b.h"
#include "util/dvpp_constexpr.h"
#include "util/dvpp_define.h"
#include "util/dvpp_error_code.h"
#include "util/dvpp_log.h"
#include "util/util.h"

namespace dvpp {
DvppErrorCode DvppBuilder910B::BuildDvppTask(
    const ge::Node& node, domi::DvppTaskDef*& dvpp_task) const {
  ge::OpDescPtr op_desc_ptr = node.GetOpDesc();
  dvpp_task->set_op_index(op_desc_ptr->GetId());
  DVPP_ENGINE_LOG_DEBUG("op[%s] ID is %ld",
                        node.GetType().c_str(), op_desc_ptr->GetId());
  return DvppErrorCode::kSuccess;
}

DvppErrorCode DvppBuilder910B::GenerateTask(const ge::Node& node,
    ge::RunContext& context, std::vector<domi::TaskDef>& tasks) {
  (void)context;
  DVPP_ENGINE_LOG_INFO("op %s[%s] run GenerateTask",
                       node.GetName().c_str(), node.GetType().c_str());
  // Check input data
  auto op_desc_ptr = node.GetOpDesc();
  DVPP_CHECK_IF_THEN_DO(op_desc_ptr == nullptr,
      DVPP_REPORT_INNER_ERR_MSG("op[%s] op_desc_ptr is nullptr",
                              node.GetType().c_str());
      return DvppErrorCode::kInputParamNull);

  domi::TaskDef task;
  task.set_type(RT_MODEL_TASK_DVPP);
  // no need to set streamID for task, GE will reallocate stream
  domi::DvppTaskDef* dvpp_task = task.mutable_dvpp_task();
  DVPP_CHECK_IF_THEN_DO(dvpp_task == nullptr,
      DVPP_REPORT_INNER_ERR_MSG("op[%s] dvpp task is nullptr",
                              node.GetType().c_str());
      return DvppErrorCode::kInputParamNull);

  // config dvpp_task
  auto error_code = BuildDvppTask(node, dvpp_task);
  DVPP_CHECK_IF_THEN_DO(error_code != DvppErrorCode::kSuccess,
      DVPP_REPORT_INNER_ERR_MSG("op[%s] BuildDvppTask failed",
                              node.GetType().c_str());
      return error_code);

  tasks.emplace_back(task);

  return DvppErrorCode::kSuccess;
}

DvppErrorCode DvppBuilder910B::SetAttrResource(
    ge::OpDescPtr& op_desc_ptr) const {
  // in 910B, device is stars, no need to set channel resource
  (void)op_desc_ptr;
  return DvppErrorCode::kSuccess;
}
} // namespace dvpp
