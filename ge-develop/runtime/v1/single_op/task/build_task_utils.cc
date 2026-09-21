/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "single_op/task/build_task_utils.h"

#include "runtime/rt.h"
#include "graph/load/model_manager/model_utils.h"
#include "graph/manager/graph_var_manager.h"
#include "graph/utils/type_utils.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/types.h"
#include "graph/utils/op_type_utils.h"

namespace ge {
std::vector<std::vector<void *>> BuildTaskUtils::GetAddresses(const OpDescPtr &op_desc,
                                                              const SingleOpModelParam &param,
                                                              const bool keep_workspace) {
  std::vector<std::vector<void *>> ret;
  ret.emplace_back(ModelUtils::GetInputDataAddrs(param.runtime_param, op_desc));
  ret.emplace_back(ModelUtils::GetOutputDataAddrs(param.runtime_param, op_desc));
  if (keep_workspace) {
    ret.emplace_back(ModelUtils::GetWorkspaceDataAddrs(param.runtime_param, op_desc));
  }
  return ret;
}

std::vector<void *> BuildTaskUtils::JoinAddresses(const std::vector<std::vector<void *>> &addresses) {
  std::vector<void *> ret;
  for (auto &address : addresses) {
    (void)ret.insert(ret.cend(), address.cbegin(), address.cend());
  }
  return ret;
}

std::vector<void *> BuildTaskUtils::GetKernelArgs(const OpDescPtr &op_desc,
                                                  const SingleOpModelParam &param) {
  const auto addresses = GetAddresses(op_desc, param);
  return JoinAddresses(addresses);
}

std::string BuildTaskUtils::InnerGetTaskInfo(const OpDescPtr &op_desc,
                                             const std::vector<const void *> &input_addrs,
                                             const std::vector<const void *> &output_addrs) {
  std::stringstream ss;
  if (op_desc != nullptr) {
    const auto op_type = op_desc->GetType();
    if ((op_type == ge::NETOUTPUT) || OpTypeUtils::IsDataNode(op_type)) {
      return ss.str();
    }
    // Conv2D IN[DT_FLOAT16 NC1HWC0[256, 128, 7, 7, 16],DT_FLOAT16 FRACTAL_Z[128, 32, 16, 16]]
    // OUT[DT_FLOAT16 NC1HWC0[256, 32, 7, 7, 16]]
    ss << op_type << " IN[";
    for (size_t idx = 0U; idx < op_desc->GetAllInputsSize(); idx++) {
      const GeTensorDescPtr &input = op_desc->MutableInputDesc(static_cast<uint32_t>(idx));
      if (input == nullptr) {
        continue;
      }
      ss << TypeUtils::DataTypeToSerialString(input->GetDataType()) << " ";
      ss << TypeUtils::FormatToSerialString(input->GetFormat());
      ss << VectorToString(input->GetShape().GetDims()) << " ";
      if (idx < input_addrs.size()) {
        ss << input_addrs[idx];
      }
      if (idx < (op_desc->GetInputsSize() - 1U)) {
        ss << ",";
      }
    }
    ss << "] OUT[";

    for (size_t idx = 0U; idx < op_desc->GetOutputsSize(); idx++) {
      const GeTensorDescPtr &output = op_desc->MutableOutputDesc(static_cast<uint32_t>(idx));
      ss << TypeUtils::DataTypeToSerialString(output->GetDataType()) << " ";
      const Format out_format = output->GetFormat();
      const GeShape &out_shape = output->GetShape();
      const auto &dims = out_shape.GetDims();
      ss << TypeUtils::FormatToSerialString(out_format);
      ss << VectorToString(dims) << " ";
      if (idx < output_addrs.size()) {
        ss << output_addrs[idx];
      }
      if (idx < (op_desc->GetOutputsSize() - 1U)) {
        ss << ",";
      }
    }
    ss << "]\n";
  }
  return ss.str();
}

std::string BuildTaskUtils::GetTaskInfo(const OpDescPtr &op_desc) {
  const std::vector<const void *> input_addrs;
  const std::vector<const void *> output_addrs;
  return InnerGetTaskInfo(op_desc, input_addrs, output_addrs);
}

std::string BuildTaskUtils::GetTaskInfo(const OpDescPtr &op_desc,
                                        const std::vector<DataBuffer> &inputs,
                                        const std::vector<DataBuffer> &outputs) {
  std::vector<const void *> input_addrs;
  std::vector<const void *> output_addrs;
  GE_CHECK_NOTNULL_EXEC(op_desc, return "");
  if (op_desc->GetAllInputsSize() == inputs.size()) {
    for (size_t i = 0U; i < inputs.size(); ++i) {
      input_addrs.push_back(inputs[i].data);
    }
  }
  if (op_desc->GetOutputsSize() == outputs.size()) {
    for (size_t i = 0U; i < outputs.size(); ++i) {
      output_addrs.push_back(outputs[i].data);
    }
  }
  return InnerGetTaskInfo(op_desc, input_addrs, output_addrs);
}

std::string BuildTaskUtils::GetTaskInfo(const hybrid::TaskContext &task_context) {
  auto &node_item = task_context.GetNodeItem();
  const auto op_desc = node_item.GetOpDesc();
  GE_CHECK_NOTNULL_EXEC(op_desc, return "");
  std::vector<const void *> input_addrs;
  std::vector<const void *> output_addrs;
  if (op_desc->GetAllInputsSize() == static_cast<uint32_t>(task_context.NumInputs())) {
    for (size_t i = 0U; i < op_desc->GetAllInputsSize(); ++i) {
      input_addrs.push_back(task_context.GetInput(static_cast<int32_t>(i))->GetData());
    }
  }
  if (op_desc->GetOutputsSize() == static_cast<uint32_t>(task_context.NumOutputs())) {
    for (size_t i = 0U; i < op_desc->GetOutputsSize(); ++i) {
      output_addrs.push_back(task_context.GetOutput(static_cast<int32_t>(i))->GetData());
    }
  }
  return InnerGetTaskInfo(op_desc, input_addrs, output_addrs);
}
}  // namespace ge
