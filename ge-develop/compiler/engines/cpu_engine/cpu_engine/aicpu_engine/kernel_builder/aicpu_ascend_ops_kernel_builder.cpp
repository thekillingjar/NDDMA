/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aicpu_ascend_ops_kernel_builder.h"
#include "../kernel_info/aicpu_kernel_info.h"
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
REGISTER_OPS_KERNEL_BUILDER(kAicpuOpsKernelInfo, AicpuAscendOpsKernelBuilder);
ge::Status AicpuAscendOpsKernelBuilder::Initialize(
    const std::map<std::string, std::string> &options) {
  (void)options;
  AICPUE_LOGI("Begin to initialize aicpu ops kernel builder");

  AICPU_CHECK_RES(Finalize());
  std::string kernel_builder = "AICPUBuilder";
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

ge::Status AicpuAscendOpsKernelBuilder::Finalize() {
  kernel_builder_map_.clear();
  return ge::SUCCESS;
}

ge::Status AicpuAscendOpsKernelBuilder::CalcOpRunningParam(ge::Node &node) {
  ge::OpDescPtr op_desc_ptr = node.GetOpDesc();
  AICPU_CHECK_NOTNULL_ERRCODE(op_desc_ptr, ErrorCode::INPUT_PARAM_NULL)
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
  const KernelBuilderPtr &kernel_builder = kernel_builder_map_["AICPUBuilder"];
  AICPU_CHECK_NOTNULL_ERRCODE(kernel_builder, ErrorCode::NONE_KERNEL_BUILDER)
  return kernel_builder->CalcOpRunningParam(node);
}

ge::Status AicpuAscendOpsKernelBuilder::GenerateTask(
    const ge::Node &node, ge::RunContext &context,
    std::vector<domi::TaskDef> &tasks) {
  ge::OpDescPtr op_desc_ptr = node.GetOpDesc();
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
  const std::string *kernel_lib_name = ge::AttrUtils::GetStr(op_desc_ptr, "opKernelLib");
  AICPU_CHECK_NOTNULL(kernel_lib_name);
  AICPUE_LOGI("Node[%s]'s kernel_lib_name is [%s].",  node.GetName().c_str(), (*kernel_lib_name).c_str());
  if (*kernel_lib_name == kAicpuKernelInfoChoice) {
    std::string ops_json_path = AicpuKernelInfo::Instance()->GetJsonPath();
    AICPU_CHECK_FALSE_EXEC(ge::AttrUtils::SetStr(op_desc_ptr, kAttrJsonPath, ops_json_path),
    AICPU_REPORT_INNER_ERR_MSG(
        "Call ge::AttrUtils::SetStr Failed to set attr[%s], op[%s].",
        kAttrJsonPath.c_str(), node.GetName().c_str());
    return ErrorCode::ADD_ATTR_FAILED)
    AICPUE_LOGI("Node[%s] set attr ops_json_path is [%s]",  node.GetName().c_str(), ops_json_path.c_str());
  }
  const KernelBuilderPtr &kernel_builder = kernel_builder_map_["AICPUBuilder"];
  AICPU_CHECK_NOTNULL_ERRCODE(kernel_builder, ErrorCode::NONE_KERNEL_BUILDER)
  return kernel_builder->GenerateTask(node, context, tasks);
}

ge::Status AicpuAscendOpsKernelBuilder::UpdateTask(const ge::Node &node,
  std::vector<domi::TaskDef> &tasks) {
  const KernelBuilderPtr &kernel_builder = kernel_builder_map_["AICPUBuilder"];
  AICPU_CHECK_NOTNULL_ERRCODE(kernel_builder, ErrorCode::NONE_KERNEL_BUILDER)
  return kernel_builder->UpdateTask(node, tasks);
}

}  // namespace aicpu
