/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aicpu_ops_kernel_builder.h"

#include "graph/utils/op_desc_utils_ex.h"
#include "base/err_msg.h"
#include "config/config_file.h"
#include "engine/base_engine.h"
#include "error_code/error_code.h"
#include "ge/ge_api_types.h"
#include "kernel_builder.h"
#include "util/constant.h"
#include "util/log.h"
#include "util/util.h"

using namespace std;
using namespace ge;

namespace aicpu {

Status AicpuOpsKernelBuilder::Initialize(const map<string, string> &options) {
  AICPUE_LOGI("Begin to initialize aicpu ops kernel builder.");
  // check options
  auto iter = options.find(ge::SOC_VERSION);

  AICPU_IF_BOOL_EXEC(iter == options.end(),
      AICPU_REPORT_INNER_ERR_MSG(
          "cannot find [%s] in param of aicpu op builder initialize function.",
          ge::SOC_VERSION.c_str());
      return INPUT_PARAM_VALID)

  AICPU_CHECK_RES(Finalize())

  string kernel_builder_str;
  string kernel_builder_config = Stringcat(engine_name_, "KernelBuilder");
  if (!ConfigFile::GetInstance().GetValue(kernel_builder_config,
                                          kernel_builder_str)) {
    AICPU_REPORT_INNER_ERR_MSG("[%s] not exist.", kernel_builder_config.c_str());
    return LOAD_KERNEL_BUILDER_FAILED;
  }

  vector<string> kernel_builders;
  ConfigFile::GetInstance().SplitValue(kernel_builder_str, kernel_builders);
  if (kernel_builders.empty()) {
    AICPU_REPORT_INNER_ERR_MSG("[%s] is empty.", kernel_builder_str.c_str());
    return NONE_KERNEL_BUILDER;
  }
  for (auto kernel_builder : kernel_builders) {
    FACTORY_KERNEL_BUILDER::FactoryType kernel_builder_ptr =
        FACTORY_KERNEL_BUILDER::Produce(kernel_builder);
    if (kernel_builder_ptr == nullptr) {
      AICPU_REPORT_INNER_ERR_MSG("create kernel builder[%s] failed",
          kernel_builder.c_str());
      return KERNEL_BUILDER_INSTANCE_FAILED;
    }
    kernel_builder_map_[kernel_builder] = kernel_builder_ptr;
    AICPU_CHECK_RES(kernel_builder_ptr->Initialize());
  }
  return SUCCESS;
}

Status AicpuOpsKernelBuilder::Finalize() {
  kernel_builder_map_.clear();
  return SUCCESS;
}

Status AicpuOpsKernelBuilder::CalcOpRunningParam(Node &node) {
  OpDescPtr op_desc_ptr = node.GetOpDesc();
  AICPU_CHECK_NOTNULL_ERRCODE(op_desc_ptr, ErrorCode::INPUT_PARAM_NULL)
  string op_type = op_desc_ptr->GetType();
  if (op_type == kFrameworkOp) {
    const string *original_type = AttrUtils::GetStr(op_desc_ptr, kOriginalType);
    AICPU_IF_BOOL_EXEC(
        (original_type == nullptr),
        AICPU_REPORT_INNER_ERR_MSG(
            "Call ge::AttrUtils::GetStr failed to get attr[%s], op[%s].",
            kOriginalType.c_str(), node.GetName().c_str());
        return ErrorCode::GET_ORIGINAL_TYPE_FAILED)
    if (original_type->empty()) {
      AICPU_REPORT_INNER_ERR_MSG("Attr[%s] is empty, op[%s].",
          kOriginalType.c_str(), node.GetName().c_str());
      return STR_IS_EMPTY;
    }
    ge::OpDescUtilsEx::SetType(op_desc_ptr, *original_type);
    op_type = *original_type;
  }
  map<string, OpFullInfo> all_op_info;
  AICPU_CHECK_RES(GetOpsInfo(all_op_info));
  string kernel_name = all_op_info[op_type].opKernelLib;
  const KernelBuilderPtr &kernel_builder = GetKernelBuilderByName(kernel_name);

  AICPU_CHECK_NOTNULL_ERRCODE(kernel_builder, ErrorCode::NONE_KERNEL_BUILDER)
  return kernel_builder->CalcOpRunningParam(node);
}

Status AicpuOpsKernelBuilder::GenerateTask(const Node &node, RunContext &context, std::vector<domi::TaskDef> &tasks) {
  OpDescPtr op_desc_ptr = node.GetOpDesc();
  AICPU_CHECK_NOTNULL_ERRCODE(op_desc_ptr, ErrorCode::INPUT_PARAM_NULL);

  // get original type
  string op_type = op_desc_ptr->GetType();
  if (op_type == kFrameworkOp) {
    const string *original_type = AttrUtils::GetStr(op_desc_ptr, kOriginalType);
    AICPU_IF_BOOL_EXEC(
        (original_type == nullptr),
        AICPU_REPORT_INNER_ERR_MSG(
            "Call ge::AttrUtils::GetStr failed to get attr[%s], op[%s].",
            kOriginalType.c_str(), node.GetName().c_str());
        return ErrorCode::GET_ORIGINAL_TYPE_FAILED)
    op_type = *original_type;
  }
  map<string, OpFullInfo> all_op_info;
  AICPU_CHECK_RES(GetOpsInfo(all_op_info));
  string kernel_name = all_op_info[op_type].opKernelLib;
  const KernelBuilderPtr &kernel_builder = GetKernelBuilderByName(kernel_name);
  AICPU_CHECK_NOTNULL_ERRCODE(kernel_builder, ErrorCode::NONE_KERNEL_BUILDER);
  return kernel_builder->GenerateTask(node, context, tasks);
}

ge::Status AicpuOpsKernelBuilder::UpdateTask(const ge::Node &node, std::vector<domi::TaskDef> &tasks) {
  OpDescPtr op_desc_ptr = node.GetOpDesc();
  AICPU_CHECK_NOTNULL_ERRCODE(op_desc_ptr, ErrorCode::INPUT_PARAM_NULL);
  string op_type = op_desc_ptr->GetType();
  if (op_type == kFrameworkOp) {
    const string *original_type = AttrUtils::GetStr(op_desc_ptr, kOriginalType);
    AICPU_IF_BOOL_EXEC((original_type == nullptr),
        AICPU_REPORT_INNER_ERR_MSG("failed to get attr[%s], op[%s].", kOriginalType.c_str(),
          node.GetName().c_str());
        return ErrorCode::GET_ORIGINAL_TYPE_FAILED)
    op_type = *original_type;
  }

  map<string, OpFullInfo> all_op_info;
  AICPU_CHECK_RES(GetOpsInfo(all_op_info));
  string kernel_name = all_op_info[op_type].opKernelLib;
  const KernelBuilderPtr &kernel_builder = GetKernelBuilderByName(kernel_name);
  AICPU_CHECK_NOTNULL_ERRCODE(kernel_builder, ErrorCode::NONE_KERNEL_BUILDER);

  return kernel_builder->UpdateTask(node, tasks);
}

Status AicpuOpsKernelBuilder::GenSingleOpRunTask(const NodePtr &node,
                                                 STR_FWK_OP_KERNEL &task,
                                                 string &task_info) {
  AICPU_CHECK_NOTNULL_ERRCODE(node, ErrorCode::INPUT_PARAM_NULL);
  OpDescPtr op_desc_ptr = node->GetOpDesc();
  AICPU_CHECK_NOTNULL_ERRCODE(op_desc_ptr, ErrorCode::INPUT_PARAM_NULL);
  string op_type = op_desc_ptr->GetType();
  map<string, OpFullInfo> all_op_info;
  AICPU_CHECK_RES(GetOpsInfo(all_op_info));
  string kernel_name = all_op_info[op_type].opKernelLib;
  const KernelBuilderPtr &kernel_builder = GetKernelBuilderByName(kernel_name);
  AICPU_CHECK_NOTNULL_ERRCODE(kernel_builder, ErrorCode::NONE_KERNEL_BUILDER);
  return kernel_builder->GenSingleOpRunTask(node, task, task_info);
}

Status AicpuOpsKernelBuilder::GenMemCopyTask(uint64_t count,
                                             STR_FWK_OP_KERNEL &task,
                                             string &task_info) {
  string kernel_name = "TFKernel";
  const KernelBuilderPtr &kernel_builder = GetKernelBuilderByName(kernel_name);
  AICPU_CHECK_NOTNULL_ERRCODE(kernel_builder, ErrorCode::NONE_KERNEL_BUILDER);
  return kernel_builder->GenMemCopyTask(count, task, task_info);
}

Status AicpuOpsKernelBuilder::GetOpsInfo(
    map<string, OpFullInfo> &all_op_info) const {
  FACTORY_ENGINE::FactoryType engine_ptr = FACTORY_ENGINE::Produce(engine_name_);
  AICPU_CHECK_NOTNULL(engine_ptr)
  AicpuOpsKernelInfoStorePtr aicpu_ops_kernel_info_store_ptr =
      engine_ptr->GetAicpuOpsKernelInfoStore();
  AICPU_CHECK_NOTNULL(aicpu_ops_kernel_info_store_ptr)
  aicpu_ops_kernel_info_store_ptr->GetAllOpsFullKernelInfo(all_op_info);
  return SUCCESS;
}

KernelBuilderPtr AicpuOpsKernelBuilder::GetKernelBuilderByName(
    const string &kernel_name) {
  string::size_type index = kernel_name.find("Kernel");
  if (index == kernel_name.npos) {
    AICPU_REPORT_INNER_ERR_MSG("Wrong kernel_name[%s], should contain 'Kernel'.",
        kernel_name.c_str());
  }
  string kernel_builder_name = kernel_name.substr(0, index) + "Builder";
  if (kernel_builder_name == "CUSTAICPUBuilder") {
    kernel_builder_name = "AICPUBuilder";
  }
  if (kernel_builder_map_.find(kernel_builder_name) != kernel_builder_map_.end()) {
    return kernel_builder_map_[kernel_builder_name];
  }
  AICPU_REPORT_INNER_ERR_MSG("kernel builder[%s] is not exist.",
      kernel_builder_name.c_str());
  return nullptr;
}
}  // namespace aicpu
