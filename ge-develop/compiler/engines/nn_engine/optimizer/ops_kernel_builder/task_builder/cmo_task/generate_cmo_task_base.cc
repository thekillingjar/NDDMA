/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ops_kernel_builder/task_builder/cmo_task/generate_cmo_task_base.h"
#include "common/fe_log.h"
#include "common/op_tensor_utils.h"
#include "graph/def_types.h"

namespace fe {
GenerateCMOTaskBase::GenerateCMOTaskBase(const ge::Node &node, TaskBuilderContext &context)
    : node_(node), context_(context) {}
GenerateCMOTaskBase::~GenerateCMOTaskBase() {}

Status GenerateCMOTaskBase::GenerateTask(std::vector<domi::TaskDef> &task_defs, const int32_t &stream_id,
                                         const std::vector<CmoAttr> &cmo_attrs) {
  (void) stream_id;
  (void) task_defs;
  (void) cmo_attrs;
  return SUCCESS;
}

Status GenerateCMOTaskBase::ParseTensorInfo(const CmoAttr &cmo_attr, TaskArgs &task_args, ge::DataType &data_type,
                                            uint64_t &source_addr, uint32_t &length_inner) const {
  int64_t tensor_size = 0;
  bool is_input = cmo_attr.object == CmoTypeObject::INPUT || cmo_attr.object == CmoTypeObject::WEIGHT;
  if (is_input) {
    auto input_size = cmo_attr.node->GetOpDesc()->GetAllInputsSize();
    if (cmo_attr.object_index >= static_cast<int32_t>(input_size)) {
      FE_LOGW("Node[%s] input object_index[%d] is out of range[%zu]",
              cmo_attr.node->GetName().c_str(), cmo_attr.object_index, input_size);
      return FAILED;
    }
    ge::GeTensorDescPtr tensor_desc_ptr = cmo_attr.node->GetOpDesc()->MutableInputDesc(cmo_attr.object_index);
    FE_CHECK_NOTNULL(tensor_desc_ptr);
    if (OpTensorUtils::CalcTensorSize(*tensor_desc_ptr, 1, tensor_size) != SUCCESS) {
      FE_LOGW("Node[%s] tensor size is out of range.", cmo_attr.node->GetName().c_str());
      return FAILED;
    }
    length_inner = static_cast<uint32_t>(tensor_size);
    data_type = tensor_desc_ptr->GetDataType();
    if (static_cast<int32_t>(task_args.input_addrs.size()) <= cmo_attr.object_index) {
      return FAILED;
    }
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
    data_type = tensor_desc_ptr->GetDataType();
    if (static_cast<int32_t>(task_args.output_addrs.size()) <= cmo_attr.object_index) {
      return FAILED;
    }
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
    if (static_cast<int32_t>(task_args.workspace_addrs.size()) <= cmo_attr.object_index) {
      return FAILED;
    }
    source_addr = ge::PtrToValue(task_args.workspace_addrs.at(cmo_attr.object_index));
  }
  return SUCCESS;
}

Status GenerateCMOTaskBase::InitCmoNodeAddrs(const ge::NodePtr &node, TaskArgs &task_args) {
  TaskBuilderAdapterPtr task_builder_adapter_ptr = nullptr;
  FE_MAKE_SHARED(task_builder_adapter_ptr = std::make_shared<TbeTaskBuilderAdapter>(*node, context_), return FAILED);
  Status status = task_builder_adapter_ptr->TaskBuilderAdapter::Init();
  if (status != SUCCESS) {
    REPORT_FE_ERROR("[GeneTask][GeneCmoTask] Init cmo node[%s] task builder adapter failed.",
                    node->GetOpDesc()->GetName().c_str());
    return status;
  }
  task_builder_adapter_ptr->GetTaskArgs(task_args);
  return SUCCESS;
}
}  // namespace fe
