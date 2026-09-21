/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "adapter/adapter_itf/task_builder_adapter.h"
#include "common/aicore_util_attr_define.h"
#include "common/fe_log.h"
#include "common/fe_type_utils.h"
#include "common/aicore_util_constants.h"
#include "common/string_utils.h"
#include "common/math_util.h"
#include "common/fe_inner_attr_define.h"
#include "common/fe_graph_common.h"
#include "common/fe_op_info_common.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/op_desc.h"
#include "graph/utils/anchor_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "framework/common/types.h"

namespace fe {
// max value of weight offset : 30 * 1024 * 1024 * 1024L.
const int64_t kMaxWeightOffset = 32212254720L;
TaskBuilderAdapter::TaskBuilderAdapter(const ge::Node &node, TaskBuilderContext &context)
    : node_(node), op_desc_(node_.GetOpDesc()), context_(context), is_unknown_graph_(false) {}

TaskBuilderAdapter::~TaskBuilderAdapter() {}

void TaskBuilderAdapter::GetTaskArgs(TaskArgs &args_info) const {
  args_info.input_addrs = input_addrs_;
  args_info.output_addrs = output_addrs_;
  args_info.workspace_addrs = workspace_addrs_;
  args_info.kernel_args_info = kernel_args_info_;
}

Status TaskBuilderAdapter::Init() {
  FE_CHECK_NOTNULL(op_desc_);
  is_unknown_graph_ = FeGraphCommon::IsNodeOfUnknownRootGraph(node_);
  FE_LOGD("Op[name=%s,type=%s]: is_unknown_graph flag is %ld.", op_desc_->GetName().c_str(),
          op_desc_->GetType().c_str(), is_unknown_graph_);
  // Init input
  Status status = InitInput();
  if (status != SUCCESS) {
    FE_LOGE("[GenTask][Init][InitInput][%s, type %s] InitInput failed.",
            op_desc_->GetName().c_str(), op_desc_->GetType().c_str());
    return status;
  }

  // Init output
  status = InitOutput();
  if (status != SUCCESS) {
    FE_LOGE("[GenTask][Init][InitOutput][%s, type %s] InitOutput failed.",
            op_desc_->GetName().c_str(), op_desc_->GetType().c_str());
    return status;
  }

  // Init workspace
  status = InitWorkspace();
  if (status != SUCCESS) {
    FE_LOGE("[GenTask][Init][InitWorkspace][%s, type %s] InitWorkspace failed.",
            op_desc_->GetName().c_str(), op_desc_->GetType().c_str());
    return status;
  }

  // Init L1 input
  status = InitInputL1Addr();
  if (status != SUCCESS) {
    FE_LOGE("[GenTask][Init][InitInputL1Addr][%s, type %s] InitInputL1Addr failed.",
            op_desc_->GetName().c_str(), op_desc_->GetType().c_str());
    return status;
  }

  if (!is_unknown_graph_) {
    // Verify weight offset.
    status = VerifyWeights();
    if (status != SUCCESS) {
      FE_LOGE("[GenTask][Init][VerifyWeights][%s, type %s] VerifyWeights failed.",
              op_desc_->GetName().c_str(), op_desc_->GetType().c_str());
      return status;
    }
  }

  return SUCCESS;
}

Status TaskBuilderAdapter::Run(domi::TaskDef &task_def) {
  (void)task_def;
  return SUCCESS;
}

void TaskBuilderAdapter::FeedArgsInfo(const domi::ArgsInfo_ArgsType &type, const domi::ArgsInfo_ArgsFormat &arg_format,
    const size_t &input_index) {
  domi::ArgsInfo arg_info;
  arg_info.set_arg_type(type);
  arg_info.set_arg_format(arg_format);
  arg_info.set_start_index(static_cast<::ascend_private::protobuf::int32>(input_index));
  uint32_t arg_size = (input_index == 0xFFFFFFFFU) ? 0U : 1U;
  arg_info.set_size(arg_size);
  kernel_args_info_.emplace_back(arg_info);
  FE_LOGD("Node[type=%s,name=%s]: Add args info[idx:%zu] finished.",
          op_desc_->GetType().c_str(), op_desc_->GetName().c_str(), input_index);
}

Status TaskBuilderAdapter::InitOutput() {
  // Verify output number.
  size_t output_num = op_desc_->GetOutputsSize();
  vector<int64_t> output_offsets = op_desc_->GetOutputOffset();
  // if output_offsets is empty, set 0 to vector
  if (output_offsets.empty()) {
    vector<int64_t> output_offset_zero(output_num, 0);
    output_offsets.swap(output_offset_zero);
    FE_LOGD("Node[type=%s,name=%s]: output_offset_size=%zu.", op_desc_->GetType().c_str(), op_desc_->GetName().c_str(),
            output_offsets.size());
  }

  if (output_num != output_offsets.size()) {
    REPORT_FE_ERROR(
        "[GenTask][InitOutput][Node %s, type %s]: output size != offset_size, output size:%zu, offset_size:%zu.",
        op_desc_->GetName().c_str(), op_desc_->GetType().c_str(), output_num, output_offsets.size());
    return FAILED;
  }
  vector<bool> output_is_addr_var;
  (void)ge::AttrUtils::GetListBool(op_desc_, ATTR_NAME_OUTPUT_IS_VAR, output_is_addr_var);
  for (size_t i = 0; i < output_num; ++i) {
    auto output_desc = op_desc_->GetOutputDesc(i);
    if (IsMemoryEmpty(output_desc)) {
      FE_LOGI("Node[type=%s,name=%s]: the output %s is memory empty.", op_desc_->GetType().c_str(),
              op_desc_->GetName().c_str(), op_desc_->GetOutputNameByIndex(i).c_str());
      std::string opt_mode;
      (void)ge::AttrUtils::GetStr(op_desc_, kAttrOptionalOutputMode, opt_mode);
      if (opt_mode == fe::kGenPlaceholder) {
        output_addrs_.push_back(reinterpret_cast<void*>(kTaskPlaceHolderAddr));
        FeedArgsInfo(domi::ArgsInfo_ArgsType_OUTPUT, domi::ArgsInfo_ArgsFormat_DIRECT_ADDR, 0xFFFFFFFFU);
      }
      continue;
    }

    int64_t output_offset = output_offsets[i];
    if (output_is_addr_var.size() > i && output_is_addr_var[i]) {
      output_addrs_.push_back(reinterpret_cast<void *>(reinterpret_cast<intptr_t>(output_offset)));
    } else {
      output_addrs_.push_back(context_.dataMemBase + output_offset);
    }
    FeedArgsInfo(domi::ArgsInfo_ArgsType_OUTPUT, domi::ArgsInfo_ArgsFormat_DIRECT_ADDR, i);
  }

  FE_LOGD("Node[type=%s,name=%s]: init output.", op_desc_->GetType().c_str(), op_desc_->GetName().c_str());
  return SUCCESS;
}

Status TaskBuilderAdapter::InitWorkspace() {
  vector<int64_t> workspace_sizes = op_desc_->GetWorkspaceBytes();
  vector<int64_t> workspace_offsets = op_desc_->GetWorkspace();
  std::vector<int64_t> mem_type_list;
  const auto has_workspace_type_list =
      ge::AttrUtils::GetListInt(op_desc_, ge::ATTR_NAME_WORKSPACE_TYPE_LIST, mem_type_list);

  if (!is_unknown_graph_) {
    if (workspace_offsets.size() != workspace_sizes.size()) {
      REPORT_FE_ERROR(
          "[GenTask][InitWorkSpace][%s, type %s]: workspaceOffsets.size()[%zu] != workspace_sizes.size()[%zu]",
          op_desc_->GetName().c_str(), op_desc_->GetType().c_str(), workspace_offsets.size(),
          workspace_sizes.size());
      return FAILED;
    }
    if (has_workspace_type_list && (mem_type_list.size() != workspace_sizes.size())) {
      REPORT_FE_ERROR(
          "[GenTask][InitWorkSpace][%s, type %s]: mem_type_list.size()[%zu] != workspace_sizes.size()[%zu]",
          op_desc_->GetName().c_str(), op_desc_->GetType().c_str(), mem_type_list.size(),
          workspace_sizes.size());
      return FAILED;
    }
  }

  size_t workspace_num = workspace_sizes.size();
  std::vector<uint32_t> aicpu_workspace_type;
  ge::AttrUtils::GetListInt(op_desc_, ge::ATTR_NAME_AICPU_WORKSPACE_TYPE, aicpu_workspace_type);
  if ((IsCustomOp(*op_desc_) || IsPrefixOpsPath(*op_desc_)) && workspace_num == aicpu_workspace_type.size()){
    for (size_t i = 0; i < workspace_num; ++i) {
      if (aicpu_workspace_type[i] == ge::AicpuWorkSpaceType::CUST_LOG) {
        workspace_sizes.erase(workspace_sizes.begin() + i);
        workspace_offsets.erase(workspace_offsets.begin() + i);
        if (has_workspace_type_list) {
          mem_type_list.erase(mem_type_list.begin() + i);
        }
        workspace_num = workspace_sizes.size();
        FE_LOGI("Node[%s][%s] custom op tiling sink remove CUST_LOG workspace[%zu]", op_desc_->GetNamePtr(),
                op_desc_->GetTypePtr(), i);
        break;
      }
    }
  }

  for (size_t i = 0; i < workspace_num; i++) {
    auto workspace_size = workspace_sizes[i];
    int64_t workspace_offset = 0;
    if (i < workspace_offsets.size()) {
      workspace_offset = workspace_offsets[i];
    }

    auto data_mem_base = context_.dataMemBase;

    if (!is_unknown_graph_) {
      FE_LOGD("Op[name=%s,type=%s]: is_unknown_graph_value flag is false.", op_desc_->GetName().c_str(),
              op_desc_->GetType().c_str());
      auto data_mem_size = context_.dataMemSize;
      if (has_workspace_type_list) {
        const auto iter = context_.mem_type_to_data_mem_size.find(mem_type_list[i]);
        if (iter == context_.mem_type_to_data_mem_size.end()) {
            REPORT_FE_ERROR("[GenTask][InitWorkSpace][%s, type %s]: can not find context_.mem_type_to_data_mem_size by "
                "mem_type_list[%lu] and value is [%ld]",
                op_desc_->GetName().c_str(), op_desc_->GetType().c_str(), i, mem_type_list[i]);
            return FAILED;
        }
        data_mem_size = iter->second;
        const auto mem_base_iter = context_.mem_type_to_data_mem_base.find(mem_type_list[i]);
        if (mem_base_iter == context_.mem_type_to_data_mem_base.end()) {
            REPORT_FE_ERROR("[GenTask][InitWorkSpace][%s, type %s]: can not find context_.mem_type_to_data_mem_base by "
                "mem_type_list[%lu]", op_desc_->GetName().c_str(), op_desc_->GetType().c_str(), i);
            return FAILED;
        }
        data_mem_base = mem_base_iter->second;
        FE_LOGD("Op[name=%s] workspace mem_type_list[%zu]: %x, data_mem_size: %llu, data_mem_base: %p.",
                op_desc_->GetName().c_str(), i, mem_type_list[i], data_mem_size, data_mem_base);
      }
      Status status = CheckOffsetAndSize(workspace_offset, static_cast<uint64_t>(workspace_size), data_mem_size);
      if (status != SUCCESS) {
        REPORT_FE_ERROR(
            "[GenTask][InitWorkSpace][%s, type %s]: Check offset and size of workspace index: %zu failed!",
            op_desc_->GetName().c_str(), op_desc_->GetType().c_str(), i);
        return status;
      }
    }

    workspace_sizes_.push_back(workspace_size);
    workspace_addrs_.push_back(workspace_size == 0 ? nullptr : data_mem_base + workspace_offset);
  }

  return SUCCESS;
}

Status TaskBuilderAdapter::InitInputL1Addr() {
  vector<int64_t> input_l1_flag;
  if (!ge::AttrUtils::GetListInt(op_desc_, ge::ATTR_NAME_OP_INPUT_L1_FLAG, input_l1_flag)) {
    FE_LOGD("Op[%s, %s] does not have OP_INPUT_L1_FLAG attribute.", op_desc_->GetName().c_str(),
            op_desc_->GetType().c_str());
    return SUCCESS;
  }
  vector<int64_t> input_l1_addrs;
  if (!ge::AttrUtils::GetListInt(op_desc_, ge::ATTR_NAME_OP_INPUT_L1_ADDR, input_l1_addrs)) {
    FE_LOGD("Op[%s, %s] does not have OP_INPUT_L1_ADDR attribute.", op_desc_->GetName().c_str(),
            op_desc_->GetType().c_str());
    return SUCCESS;
  }

  if (input_l1_flag.empty() || input_l1_addrs.empty()) {
    FE_LOGD("The vector of op_input_l1_flag or op_input_l1_addrs is empty.");
    return SUCCESS;
  }
  if (input_l1_flag.size() != input_l1_addrs.size()) {
    FE_LOGD("Size of op_input_l1_flag and op_input_l1_addrs is not equal.");
    return SUCCESS;
  }

  FE_LOGD("The value of OpInputL1Flag and OpInputL1Addrs of op[%s, %s] is [%s] and [%s].", op_desc_->GetName().c_str(),
          op_desc_->GetType().c_str(), StringUtils::IntegerVecToString(input_l1_flag).c_str(),
          StringUtils::IntegerVecToString(input_l1_addrs).c_str());

  for (size_t i = 0; i < input_l1_flag.size(); i++) {
    if (input_l1_flag[i] >= 0) {
      input_l1_addrs_.push_back(reinterpret_cast<void *>(reinterpret_cast<intptr_t>(input_l1_addrs[i])));
    }
  }

  return SUCCESS;
}

Status TaskBuilderAdapter::VerifyWeights() {
  // Verify weight offset.
  vector<ge::ConstGeTensorPtr> weights = ge::OpDescUtils::GetWeights(node_);
  for (const ge::ConstGeTensorPtr& weight : weights) {
    int64_t weight_offset = 0;
    if (ge::TensorUtils::GetDataOffset(weight->GetTensorDesc(), weight_offset) != ge::GRAPH_SUCCESS) {
      REPORT_FE_ERROR("[GenTask][VerifyWeights][%s, type %s]: Get weight offset failed.",
                      op_desc_->GetName().c_str(), op_desc_->GetType().c_str());
      return FAILED;
    }

    uint64_t weight_size = ge::TensorUtils::GetWeightSize(weight);
    Status status = CheckOffsetAndSize(weight_offset, weight_size, context_.weightMemSize);
    if (status != SUCCESS) {
      REPORT_FE_ERROR("[GenTask][VerifyWeights][%s, type %s]: Check offset and size of weight failed.",
                      op_desc_->GetName().c_str(), op_desc_->GetType().c_str());
      return status;
    }
  }

  return SUCCESS;
}

Status TaskBuilderAdapter::CheckOffsetAndSize(int64_t offset, uint64_t space_size, uint64_t total_size) {
  if (offset < 0) {
    REPORT_FE_ERROR("[GenTask][Init][Check] offset should not be less than 0. offset[%ld]", offset);
    return FAILED;
  }
  FE_UINT64_ADDCHECK(static_cast<uint64_t>(offset), space_size);
  if (static_cast<uint64_t>(offset) + space_size > total_size) {
    REPORT_FE_ERROR(
        "[GenTask][Init][Check] offset[%ld] + size[%lu] should not be greater than total_size[%lu]",
        offset, space_size, total_size);
    return FAILED;
  }

  return SUCCESS;
}
}  // namespace fe
