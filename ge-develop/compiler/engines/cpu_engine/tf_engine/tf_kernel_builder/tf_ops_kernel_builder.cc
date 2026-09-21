/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "util/log.h"
#include "error_code/error_code.h"
#include "base/err_msg.h"
#include "ge/ge_api_types.h"
#include "register/ops_kernel_builder_registry.h"
#include "common/util/util.h"
#include "common/config/config_file.h"
#include "tf_kernel_builder/tf_ops_kernel_builder.h"
#include "graph/utils/op_desc_utils_ex.h"
#include "tf_kernel_info/tf_kernel_info.h"

namespace aicpu {
REGISTER_OPS_KERNEL_BUILDER(kTfOpsKernelInfo, AicpuTfOpsKernelBuilder);

ge::Status AicpuTfOpsKernelBuilder::Initialize(const std::map<std::string, std::string> &options) {
  (void)options;
  AICPUE_LOGI("Begin to initialize aicpu ops kernel builder.");
  AICPU_CHECK_RES(Finalize());

  std::string kernel_builder_str = "TFBuilder";
  FACTORY_KERNEL_BUILDER::FactoryType kernel_builder_ptr = FACTORY_KERNEL_BUILDER::Produce(kernel_builder_str);
  if (kernel_builder_ptr == nullptr) {
    AICPU_REPORT_INNER_ERR_MSG("[%s] instantiate failed", kernel_builder_str.c_str());
    return KERNEL_BUILDER_INSTANCE_FAILED;
  }
  kernel_builder_map_[kernel_builder_str] = kernel_builder_ptr;
  AICPU_CHECK_RES(kernel_builder_ptr->Initialize());
  return ge::SUCCESS;
}

ge::Status AicpuTfOpsKernelBuilder::Finalize() {
  kernel_builder_map_.clear();
  return ge::SUCCESS;
}

ge::Status AicpuTfOpsKernelBuilder::CalcOpRunningParam(ge::Node &node) {
  ge::OpDescPtr op_desc_ptr = node.GetOpDesc();
  AICPU_CHECK_NOTNULL_ERRCODE(op_desc_ptr, ErrorCode::INPUT_PARAM_NULL);
  std::string op_type = op_desc_ptr->GetType();
  if (op_type == kFrameworkOp) {
    const std::string *original_type = ge::AttrUtils::GetStr(op_desc_ptr, kOriginalType);
    AICPU_IF_BOOL_EXEC((original_type == nullptr),
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
  const KernelBuilderPtr &kernel_builder = kernel_builder_map_["TFBuilder"];
  AICPU_CHECK_NOTNULL_ERRCODE(kernel_builder, ErrorCode::NONE_KERNEL_BUILDER)
  return kernel_builder->CalcOpRunningParam(node);
}

ge::Status AicpuTfOpsKernelBuilder::GenerateTask(const ge::Node &node, ge::RunContext &context,
                                                 vector<domi::TaskDef> &tasks) {
  ge::OpDescPtr op_desc_ptr = node.GetOpDesc();
  AICPU_CHECK_NOTNULL_ERRCODE(op_desc_ptr, ErrorCode::INPUT_PARAM_NULL);
  // get original type
  std::string op_type = op_desc_ptr->GetType();
  if (op_type == kFrameworkOp) {
    const std::string *original_type = ge::AttrUtils::GetStr(op_desc_ptr, kOriginalType);
    AICPU_IF_BOOL_EXEC((original_type == nullptr),
        AICPU_REPORT_INNER_ERR_MSG(
            "Call ge::AttrUtils::GetStr failed to get attr[%s], op[%s].",
            kOriginalType.c_str(), op_desc_ptr->GetName().c_str());
        return ErrorCode::GET_ORIGINAL_TYPE_FAILED);
    op_type = *original_type;
  }
  string ops_json_path = TfKernelInfo::Instance()->GetJsonPath();
  AICPU_CHECK_FALSE_EXEC(ge::AttrUtils::SetStr(op_desc_ptr, kAttrJsonPath, ops_json_path),
      AICPU_REPORT_INNER_ERR_MSG(
          "Call ge::AttrUtils::SetStr Failed to set attr[%s], op[%s].",
          kAttrJsonPath.c_str(), node.GetName().c_str());
      return ErrorCode::ADD_ATTR_FAILED)
  AICPUE_LOGI("Node[%s] attr ops_json_path is [%s].", node.GetName().c_str(), ops_json_path.c_str());
  const KernelBuilderPtr &kernel_builder = kernel_builder_map_["TFBuilder"];
  AICPU_CHECK_NOTNULL_ERRCODE(kernel_builder, ErrorCode::NONE_KERNEL_BUILDER);
  return kernel_builder->GenerateTask(node, context, tasks);
}

ge::Status AicpuTfOpsKernelBuilder::UpdateTask(const ge::Node &node, std::vector<domi::TaskDef> &tasks) {
  const KernelBuilderPtr &kernel_builder = kernel_builder_map_["TFBuilder"];
  AICPU_CHECK_NOTNULL_ERRCODE(kernel_builder, ErrorCode::NONE_KERNEL_BUILDER);
  return kernel_builder->UpdateTask(node, tasks);
}

ge::Status AicpuTfOpsKernelBuilder::GenSingleOpRunTask(const ge::NodePtr &node,
                                                       STR_FWK_OP_KERNEL &task,
                                                       std::string &task_info) {
  ge::OpDescPtr op_desc_ptr = node->GetOpDesc();
  AICPU_CHECK_NOTNULL_ERRCODE(op_desc_ptr, ErrorCode::INPUT_PARAM_NULL);
  std::string op_type = op_desc_ptr->GetType();
  const KernelBuilderPtr &kernel_builder = kernel_builder_map_["TFBuilder"];
  AICPU_CHECK_NOTNULL_ERRCODE(kernel_builder, ErrorCode::NONE_KERNEL_BUILDER);
  return kernel_builder->GenSingleOpRunTask(node, task, task_info);
}

ge::Status AicpuTfOpsKernelBuilder::GenMemCopyTask(uint64_t count,
                                                   STR_FWK_OP_KERNEL &task,
                                                   std::string &task_info) {
  const KernelBuilderPtr &kernel_builder = kernel_builder_map_["TFBuilder"];
  AICPU_CHECK_NOTNULL_ERRCODE(kernel_builder, ErrorCode::NONE_KERNEL_BUILDER);
  return kernel_builder->GenMemCopyTask(count, task, task_info);
}
} // namespace aicpu
