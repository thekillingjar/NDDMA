/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ops_kernel_builder/task_builder/cmo_task/generate_cmo_invalid_task.h"
#include "common/fe_log.h"
#include "graph/ge_tensor.h"
#include "graph/def_types.h"
#include "common/op_tensor_utils.h"

namespace fe {
GenerateCMOInvalidTask::GenerateCMOInvalidTask(const ge::Node &node, TaskBuilderContext &context)
    : GenerateCMOTaskBase(node, context) {}
GenerateCMOInvalidTask::~GenerateCMOInvalidTask() {}

Status GenerateCMOInvalidTask::GenerateTask(std::vector<domi::TaskDef> &task_defs, const int32_t &stream_id,
                                            const std::vector<CmoAttr> &cmo_attrs) {
  for (auto &cmo_attr : cmo_attrs) {
    TaskArgs task_args;
    if (InitCmoNodeAddrs(cmo_attr.node, task_args) != SUCCESS) {
      FE_LOGW("[GenTask][GenInvalidTask] Failed to initialize node [%s] addresses.", cmo_attr.node->GetName().c_str());
      return FAILED;
    }
    domi::TaskDef task_def;
    task_def.set_stream_id(stream_id);
    task_def.set_type(RT_MODEL_TASK_CMO);
    domi::CmoTaskDef *cmo_task_def = task_def.mutable_cmo_task();
    if (cmo_task_def == nullptr) {
      FE_LOGW("Failed to create cmo task definition for node [%s].", node_.GetName().c_str());
      return FAILED;
    }
    rtCMOType cmoType = rtCMOType::rtCMOInvalid;
    cmo_task_def->set_cmo_type(static_cast<uint32_t>(cmoType));
    int64_t complex_cmo_id_reuse = 0;
    // gen cmo id
    uint32_t cmo_id = 0;
    if (GetComplexCmoId(cmo_attr, complex_cmo_id_reuse)) {
      cmo_id = static_cast<uint32_t>(complex_cmo_id_reuse & 0x00000000ffffffff);
    } else {
      cmo_id = static_cast<uint32_t>(CMOIdGenStrategy::Instance().GenerateCMOId(node_));
    }
    if (cmo_id == 0) {
      FE_LOGW("Failed to generate cmo id for mode [%s], cmo task not launched.", node_.GetName().c_str());
      return FAILED;
    }
    cmo_task_def->set_logic_id(cmo_id);
    ge::DataType data_type = ge::DT_UNDEFINED;
    uint32_t length_inner = 0;
    uint64_t source_addr;
    int64_t complex_cmo_id = (static_cast<int64_t>(cmoType) << 32) + cmo_id;
    Status status = ParseTensorInfo(cmo_attr, task_args, data_type, complex_cmo_id, source_addr, length_inner);
    if (status != SUCCESS) {
      FE_LOGW("Failed to generate cmo task definition for node [%s].", node_.GetName().c_str());
      return FAILED;
    }
    FE_LOGD("Generate invalid cmo task id[%u] for node[%s] success.", cmo_id, node_.GetName().c_str());

    // low 4bit: cmo type; high 4bit: data type
    // invalid: 0x8
    uint8_t op_code = 0x8;
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

Status GenerateCMOInvalidTask::ParseTensorInfo(const CmoAttr &cmo_attr, const TaskArgs &task_args,
                                               ge::DataType &data_type, int64_t &complex_cmo_id, uint64_t &source_addr,
                                               uint32_t &length_inner) const {
  int64_t tensor_size = 0;
  if (cmo_attr.object == CmoTypeObject::INPUT || cmo_attr.object == CmoTypeObject::WEIGHT) {
    auto input_size = cmo_attr.node->GetOpDesc()->GetAllInputsSize();
    if (cmo_attr.object_index >= static_cast<int32_t>(input_size)) {
      FE_LOGW("Node[%s] input object_index[%d] is out of range[%zu]",
              cmo_attr.node->GetName().c_str(), cmo_attr.object_index, input_size);
      return FAILED;
    }
    ge::GeTensorDescPtr tensor_desc_ptr = cmo_attr.node->GetOpDesc()->MutableInputDesc(cmo_attr.object_index);
    FE_CHECK_NOTNULL(tensor_desc_ptr);
    if (OpTensorUtils::CalcTensorSize(*tensor_desc_ptr, 1, tensor_size) != SUCCESS || tensor_size > UINT32_MAX) {
      FE_LOGW("Node[%s] tensor size is out of range.", cmo_attr.node->GetName().c_str());
      return FAILED;
    }
    length_inner = static_cast<uint32_t>(tensor_size);
    (void)ge::AttrUtils::SetInt(tensor_desc_ptr, "_complex_cmo_id", complex_cmo_id);
    data_type = tensor_desc_ptr->GetDataType();
    source_addr = ge::PtrToValue(task_args.input_addrs.at(cmo_attr.object_index));
  } else if (cmo_attr.object == CmoTypeObject::OUTPUT) {
    auto output_size = cmo_attr.node->GetOpDesc()->GetAllOutputsDescSize();
    if (cmo_attr.object_index >= static_cast<int32_t>(output_size)) {
      FE_LOGW("Node[%s] output object_index[%d] is out of range[%d]",
              cmo_attr.node->GetName().c_str(), cmo_attr.object_index, output_size);
      return FAILED;
    }
    ge::GeTensorDescPtr tensor_desc_ptr = cmo_attr.node->GetOpDesc()->MutableOutputDesc(cmo_attr.object_index);
    if (OpTensorUtils::CalcTensorSize(*tensor_desc_ptr, 1, tensor_size) != SUCCESS || tensor_size > UINT32_MAX) {
      FE_LOGW("Node[%s] tensor size is out of range.", cmo_attr.node->GetName().c_str());
      return FAILED;
    }
    length_inner = static_cast<uint32_t>(tensor_size);
    (void)ge::AttrUtils::SetInt(tensor_desc_ptr, "_complex_cmo_id", complex_cmo_id);
    data_type = tensor_desc_ptr->GetDataType();
    source_addr = ge::PtrToValue(task_args.output_addrs.at(cmo_attr.object_index));
  } else {
    std::vector<int64_t> workspace_bytes = cmo_attr.node->GetOpDesc()->GetWorkspaceBytes();
    if (cmo_attr.object_index >= static_cast<int32_t>(workspace_bytes.size()) ||
        workspace_bytes[cmo_attr.object_index] > UINT32_MAX) {
      FE_LOGW("Node[%s] workspace object_index[%d] is out of range[%zu]",
              cmo_attr.node->GetName().c_str(), cmo_attr.object_index, workspace_bytes.size());
      return FAILED;
    }
    length_inner = static_cast<uint32_t>(workspace_bytes[cmo_attr.object_index]);
    data_type = ge::DT_INT8;
    (void)ge::AttrUtils::SetInt(cmo_attr.node->GetOpDesc(),
                                "_worksapce_" + std::to_string(cmo_attr.object_index) + "_complex_cmo_id",
                                complex_cmo_id);
    source_addr = ge::PtrToValue(task_args.workspace_addrs.at(cmo_attr.object_index));
  }
  return SUCCESS;
}

bool GenerateCMOInvalidTask::GetComplexCmoId(const CmoAttr &cmo_attr, int64_t &complex_cmo_id) const {
  ge::NodePtr node = cmo_attr.node;
  if (cmo_attr.object == CmoTypeObject::INPUT || cmo_attr.object == CmoTypeObject::WEIGHT) {
    auto input_size = node->GetOpDesc()->GetAllInputsSize();
    if (cmo_attr.object_index >= static_cast<int32_t>(input_size)) {
      FE_LOGW("Node[%s] input object_index[%d] is out of range[%zu]",
              cmo_attr.node->GetName().c_str(), cmo_attr.object_index, input_size);
      return false;
    }
    ge::GeTensorDescPtr tensor_desc_ptr = node->GetOpDesc()->MutableInputDesc(cmo_attr.object_index);
    FE_CHECK_NOTNULL(tensor_desc_ptr);
    if (!ge::AttrUtils::HasAttr(tensor_desc_ptr, "_complex_cmo_id")) {
      return false;
    }
    (void)ge::AttrUtils::GetInt(tensor_desc_ptr, "_complex_cmo_id", complex_cmo_id);
    FE_LOGD("Get invalid task complex cmo id[%ld] from node[%s] success.", complex_cmo_id, node->GetName().c_str());
  } else if (cmo_attr.object == CmoTypeObject::OUTPUT) {
    auto output_size = node->GetOpDesc()->GetAllOutputsDescSize();
    if (cmo_attr.object_index >= static_cast<int32_t>(output_size)) {
      FE_LOGW("Node[%s] output object_index[%d] is out of range[%d]",
              cmo_attr.node->GetName().c_str(), cmo_attr.object_index, output_size);
      return false;
    }
    ge::GeTensorDescPtr tensor_desc_ptr = node->GetOpDesc()->MutableOutputDesc(cmo_attr.object_index);
    if (!ge::AttrUtils::HasAttr(tensor_desc_ptr, "_complex_cmo_id")) {
      return false;
    }
    (void)ge::AttrUtils::GetInt(tensor_desc_ptr, "_complex_cmo_id", complex_cmo_id);
    FE_LOGD("Get invalid task complex cmo id[%ld] from node[%s] success.", complex_cmo_id, node->GetName().c_str());
  } else {
    auto workspace_bytes = node->GetOpDesc()->GetWorkspaceBytes();
    if (cmo_attr.object_index >= static_cast<int32_t>(workspace_bytes.size())) {
      FE_LOGW("Node[%s] workspace object_index[%d] is out of range[%zu]",
              cmo_attr.node->GetName().c_str(), cmo_attr.object_index, workspace_bytes.size());
      return false;
    }
    if (!ge::AttrUtils::HasAttr(
        node->GetOpDesc(), "_worksapce_" + std::to_string(cmo_attr.object_index) + "_complex_cmo_id")) {
      return false;
    }
    (void)ge::AttrUtils::GetInt(node->GetOpDesc(),
                                "_worksapce_" + std::to_string(cmo_attr.object_index) + "_complex_cmo_id",
                                complex_cmo_id);
    FE_LOGD("Get invalid task complex cmo id[%ld] from node[%s] success.", complex_cmo_id, node->GetName().c_str());
  }
  return true;
}
}
