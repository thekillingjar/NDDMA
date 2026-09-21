/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ops_kernel_builder/task_builder/cmo_task/generate_cmo_barrier_task.h"
#include "common/fe_log.h"

namespace fe {
const uint8_t kMaxBarrierTaskNum = 6;
GenerateCMOBarrierTask::GenerateCMOBarrierTask(const ge::Node &node, TaskBuilderContext &context)
    : GenerateCMOTaskBase(node, context) {}
GenerateCMOBarrierTask::~GenerateCMOBarrierTask() {}

Status GenerateCMOBarrierTask::GenerateTask(std::vector<domi::TaskDef> &task_defs, const int32_t &stream_id,
                                            const std::vector<CmoAttr> &cmo_attrs) {
  domi::TaskDef task_def;
  task_def.set_stream_id(stream_id);
  task_def.set_type(RT_MODEL_TASK_BARRIER);
  domi::CmoBarrierTaskDef *cmo_task_def = task_def.mutable_cmo_barrier_task();
  if (cmo_task_def == nullptr) {
    FE_LOGW("Create cmo task def for node[%s] failed.", node_.GetName().c_str());
    return FAILED;
  }
  // get cmo id
  uint8_t cmoNum = 0;
  std::vector<uint32_t> cmo_ids;
  for (auto &cmo_attr : cmo_attrs) {
    int64_t complex_cmo_id = 0;
    if (GenerateCMOId(cmo_attr, complex_cmo_id) != SUCCESS) {
      FE_LOGW("Generate barrier cmo task of node[%s] failed.", node_.GetName().c_str());
      return FAILED;
    }
    uint32_t cmo_id = static_cast<uint32_t>(complex_cmo_id & 0x00000000ffffffff);
    uint16_t cmo_type = static_cast<uint16_t>((complex_cmo_id & 0xffffffff00000000) >> 32);
    if (cmo_id == 0) {
      continue;
    }
    FE_LOGD("Generate barrier cmo task[id:%u, type:%u] before node[%s].", cmo_id, cmo_type, node_.GetName().c_str());
    domi::CmoBarrierInfoDef *barrier_info_ptr = cmo_task_def->add_barrier_info();
    FE_CHECK_NOTNULL(barrier_info_ptr);
    barrier_info_ptr->set_cmo_type(cmo_type);
    barrier_info_ptr->set_logic_id(cmo_id);
    cmo_ids.emplace_back(cmo_id);
    cmoNum++;
    if (cmoNum >= kMaxBarrierTaskNum) {
      break;
    }
  }
  if (cmoNum == 0) {
    // if set set_logic_id_num 0, runtime excute fail
    FE_LOGD("all barrier task has been continued before node[%s]", node_.GetName().c_str());
    return SUCCESS;
  }
  for (auto cmo_id : cmo_ids) {
    CMOIdGenStrategy::Instance().UpdateReuseMap(node_.GetOpDesc()->GetId(), cmo_id);
  }
  cmo_task_def->set_logic_id_num(cmoNum);
  task_defs.push_back(task_def);
  return SUCCESS;
}

Status GenerateCMOBarrierTask::GenerateCMOId(const CmoAttr &cmo_attr, int64_t &complex_cmo_id) const {
  ge::NodePtr pre_node = cmo_attr.node;
  if (cmo_attr.object == CmoTypeObject::INPUT || cmo_attr.object == CmoTypeObject::WEIGHT) {
    auto input_size = pre_node->GetOpDesc()->GetAllInputsSize();
    if (cmo_attr.object_index >= static_cast<int32_t>(input_size)) {
      FE_LOGW("Node[%s] input object_index[%d] is out of range[%zu]",
              cmo_attr.node->GetName().c_str(), cmo_attr.object_index, input_size);
      return FAILED;
    }
    ge::GeTensorDescPtr tensor_desc_ptr = pre_node->GetOpDesc()->MutableInputDesc(cmo_attr.object_index);
    (void)ge::AttrUtils::GetInt(tensor_desc_ptr, "_complex_cmo_id", complex_cmo_id);
    FE_LOGD("Get complex cmo id[%ld] from node[%s] success.", complex_cmo_id, pre_node->GetName().c_str());
  } else if (cmo_attr.object == CmoTypeObject::OUTPUT) {
    auto output_size = pre_node->GetOpDesc()->GetAllOutputsDescSize();
    if (cmo_attr.object_index >= static_cast<int32_t>(output_size)) {
      FE_LOGW("Node[%s] output object_index[%d] is out of range[%d]",
              cmo_attr.node->GetName().c_str(), cmo_attr.object_index, output_size);
      return FAILED;
    }
    ge::GeTensorDescPtr tensor_desc_ptr = pre_node->GetOpDesc()->MutableOutputDesc(cmo_attr.object_index);
    (void)ge::AttrUtils::GetInt(tensor_desc_ptr, "_complex_cmo_id", complex_cmo_id);
    FE_LOGD("Get complex cmo id[%ld] from node[%s] success.", complex_cmo_id, pre_node->GetName().c_str());
  } else {
    auto workspace_bytes = pre_node->GetOpDesc()->GetWorkspaceBytes();
    if (cmo_attr.object_index >= static_cast<int32_t>(workspace_bytes.size())) {
      FE_LOGW("Node[%s] workspace object_index[%d] is out of range[%zu]",
              cmo_attr.node->GetName().c_str(), cmo_attr.object_index, workspace_bytes.size());
      return FAILED;
    }
    (void)ge::AttrUtils::GetInt(pre_node->GetOpDesc(),
                                "_worksapce_" + std::to_string(cmo_attr.object_index) + "_complex_cmo_id",
                                complex_cmo_id);
    FE_LOGD("Get complex cmo id[%ld] from node[%s] success.", complex_cmo_id, pre_node->GetName().c_str());
  }
  return SUCCESS;
}
}
