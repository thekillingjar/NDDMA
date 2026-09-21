/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hostcpu_ops_kernel_builder.h"
#include <memory>
#include <vector>
#include "common/config/config_file.h"
#include "base/err_msg.h"
#include "common/util/util.h"
#include "error_code/error_code.h"
#include "ge/ge_api_types.h"
#include "register/ops_kernel_builder_registry.h"
#include "graph/utils/op_desc_utils_ex.h"

namespace aicpu {
REGISTER_OPS_KERNEL_BUILDER(kHostCpuOpsKernelInfo, HostCpuOpsKernelBuilder);
ge::Status HostCpuOpsKernelBuilder::Initialize(
    const std::map<std::string, std::string> &options) {
  (void)options;
  AICPUE_LOGI("Begin to initialize host cpu ops kernel builder");

  AICPU_CHECK_RES(Finalize());

  std::string kernel_builder = "HOSTCPUBuilder";
  FACTORY_KERNEL_BUILDER::FactoryType kernel_builder_ptr =
      FACTORY_KERNEL_BUILDER::Produce(kernel_builder);
  if (kernel_builder_ptr == nullptr) {
    AICPU_REPORT_INNER_ERR_MSG("Crate %s failed.", kernel_builder.c_str());
    return KERNEL_BUILDER_INSTANCE_FAILED;
  }
  kernel_builder_map_[kernel_builder] = kernel_builder_ptr;
  AICPU_CHECK_RES(kernel_builder_ptr->Initialize());
  return ge::SUCCESS;
}

ge::Status HostCpuOpsKernelBuilder::Finalize() {
  kernel_builder_map_.clear();
  return ge::SUCCESS;
}

ge::Status HostCpuOpsKernelBuilder::CalcOpRunningParam(ge::Node &node) {
  ge::OpDescPtr op_desc_ptr = node.GetOpDesc();
  AICPU_CHECK_NOTNULL_ERRCODE(op_desc_ptr, ErrorCode::INPUT_PARAM_NULL);
  std::string op_type = op_desc_ptr->GetType();
  if (op_type == kFrameworkOp) {
    const std::string *original_type = ge::AttrUtils::GetStr(op_desc_ptr, kOriginalType);
    AICPU_IF_BOOL_EXEC(
        (original_type == nullptr),
        AICPU_REPORT_INNER_ERR_MSG("Get attr[%s] of op[%s] failed.",
            kOriginalType.c_str(), node.GetName().c_str());
        return ErrorCode::GET_ORIGINAL_TYPE_FAILED)
    if (original_type->empty()) {
      AICPU_REPORT_INNER_ERR_MSG("Attr[%s] of op[%s] is empty.",
                kOriginalType.c_str(), node.GetName().c_str());
      return STR_IS_EMPTY;
    }
    ge::OpDescUtilsEx::SetType(op_desc_ptr, *original_type);
    op_type = *original_type;
  }
  const KernelBuilderPtr &kernel_builder = kernel_builder_map_["HOSTCPUBuilder"];
  AICPU_CHECK_NOTNULL_ERRCODE(kernel_builder, ErrorCode::NONE_KERNEL_BUILDER);
  return kernel_builder->CalcOpRunningParam(node);
}

ge::Status HostCpuOpsKernelBuilder::GenerateTask(
    const ge::Node &ge_node, ge::RunContext &context,
    std::vector<domi::TaskDef> &tasks) {
  ge::OpDescPtr op_desc_ptr = ge_node.GetOpDesc();
  AICPU_CHECK_NOTNULL_ERRCODE(op_desc_ptr, ErrorCode::INPUT_PARAM_NULL);

  // get original type
  std::string op_type = op_desc_ptr->GetType();
  if (op_type == kFrameworkOp) {
    const std::string *original_type = ge::AttrUtils::GetStr(op_desc_ptr, kOriginalType);
    AICPU_IF_BOOL_EXEC(
        (original_type == nullptr),
         AICPU_REPORT_INNER_ERR_MSG("Get attr[%s] of op[%s] failed.",
            kOriginalType.c_str(), op_desc_ptr->GetName().c_str());
        return ErrorCode::GET_ORIGINAL_TYPE_FAILED)
    op_type = *original_type;
  }

  const KernelBuilderPtr &kernel_builder = kernel_builder_map_["HOSTCPUBuilder"];
  AICPU_CHECK_NOTNULL_ERRCODE(kernel_builder, ErrorCode::NONE_KERNEL_BUILDER);
  return kernel_builder->GenerateTask(ge_node, context, tasks);
}

ge::Status HostCpuOpsKernelBuilder::UpdateTask(const ge::Node &node,
  std::vector<domi::TaskDef> &tasks) {
  const KernelBuilderPtr &kernel_builder = kernel_builder_map_["HOSTCPUBuilder"];
  AICPU_CHECK_NOTNULL_ERRCODE(kernel_builder, ErrorCode::NONE_KERNEL_BUILDER);
  return kernel_builder->UpdateTask(node, tasks);
}
}  // namespace aicpu
