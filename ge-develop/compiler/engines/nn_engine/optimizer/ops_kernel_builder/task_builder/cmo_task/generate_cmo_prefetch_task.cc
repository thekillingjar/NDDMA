/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ops_kernel_builder/task_builder/cmo_task/generate_cmo_prefetch_task.h"
#include "common/fe_log.h"

namespace fe {
GenerateCMOPrefetchTask::GenerateCMOPrefetchTask(const ge::Node &node, TaskBuilderContext &context)
    : GenerateCMOTaskBase(node, context) {}
GenerateCMOPrefetchTask::~GenerateCMOPrefetchTask() {}


Status GenerateCMOPrefetchTask::GenerateTask(std::vector<domi::TaskDef> &task_defs, const int32_t &stream_id,
                                             const std::vector<CmoAttr> &cmo_attrs) {
  for (auto &cmo_attr : cmo_attrs) {
    TaskArgs task_args;
    if (InitCmoNodeAddrs(cmo_attr.node, task_args) != SUCCESS) {
      FE_LOGW("[GenTask][GenPrefetchTask] Init node[%s] addrs failed.", cmo_attr.node->GetName().c_str());
      return FAILED;
    }
    domi::TaskDef task_def;
    task_def.set_stream_id(stream_id);
    task_def.set_type(RT_MODEL_TASK_CMO);
    domi::CmoTaskDef *cmo_task_def = task_def.mutable_cmo_task();
    FE_CHECK(cmo_task_def == nullptr, FE_LOGW("Failed to create cmo task definition for node [%s].", node_.GetName().c_str()),
             return FAILED);
    cmo_task_def->set_cmo_type(static_cast<uint32_t>(rtCMOType::rtCMOPrefetch));
    ge::DataType data_type = ge::DT_UNDEFINED;
    uint32_t length_inner = 0;
    uint64_t source_addr;
    Status status = ParseTensorInfo(cmo_attr, task_args, data_type, source_addr, length_inner);
    if (status != SUCCESS) {
      FE_LOGD("Not generate some cmo task def for node[%s].", node_.GetName().c_str());
      continue;
    }
    // gen cmo id
    uint32_t cmo_id = static_cast<uint32_t>(CMOIdGenStrategy::Instance().GenerateCMOId(node_));
    if (cmo_id == 0) {
      FE_LOGW("Failed to generate cmo id for mode [%s], cmo task not launched.", node_.GetName().c_str());
      return FAILED;
    }
    FE_LOGD("Generate prefetch cmo task id[%u] for node[%s] success.", cmo_id, node_.GetName().c_str());
    cmo_task_def->set_logic_id(cmo_id);
    // low 4bit: cmo type; high 4bit: data type
    // prefetch: 0x6
    uint8_t op_code = 0x6;
    if (DATA_TYPE_CODE.count(data_type) == 0) {
      op_code += 0xf0;
    } else {
      op_code += (DATA_TYPE_CODE.at(data_type) << 4);
    }
    cmo_task_def->set_op_code(op_code);
    cmo_task_def->set_qos(0);
    cmo_task_def->set_part_id(0);
    cmo_task_def->set_pmg(0);
    cmo_task_def->set_num_inner(1);
    cmo_task_def->set_num_outer(1);
    cmo_task_def->set_length_inner(length_inner);
    cmo_task_def->set_source_addr(source_addr);
    cmo_task_def->set_strider_outer(0);
    cmo_task_def->set_strider_inner(0);

    task_defs.push_back(task_def);
  }
  return SUCCESS;
}
}
