/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_SINGLE_OP_TASK_BUILD_TASK_UTILS_H_
#define GE_SINGLE_OP_TASK_BUILD_TASK_UTILS_H_

#include <vector>
#include <sstream>

#include "graph/op_desc.h"
#include "single_op/single_op.h"
#include "single_op/single_op_model.h"
#include "hybrid/node_executor/task_context.h"

namespace ge {
class BuildTaskUtils {
 public:
  static std::vector<std::vector<void *>> GetAddresses(const OpDescPtr &op_desc,
                                                       const SingleOpModelParam &param,
                                                       const bool keep_workspace = true);
  static std::vector<void *> JoinAddresses(const std::vector<std::vector<void *>> &addresses);
  static std::vector<void *> GetKernelArgs(const OpDescPtr &op_desc, const SingleOpModelParam &param);
  static std::string InnerGetTaskInfo(const OpDescPtr &op_desc,
                                      const std::vector<const void *> &input_addrs,
                                      const std::vector<const void *> &output_addrs);
  static std::string GetTaskInfo(const OpDescPtr &op_desc);
  static std::string GetTaskInfo(const OpDescPtr &op_desc,
                                 const std::vector<DataBuffer> &inputs,
                                 const std::vector<DataBuffer> &outputs);
  static std::string GetTaskInfo(const hybrid::TaskContext& task_context);

  template<typename T>
  static std::string VectorToString(const std::vector<T> &values) {
    std::stringstream ss;
    ss << '[';
    const auto size = values.size();
    for (size_t i = 0U; i < size; ++i) {
      ss << values[i];
      if (i != (size - 1U)) {
        ss << ", ";
      }
    }
    ss << ']';
    return ss.str();
  }
};
}  // namespace ge
#endif  // GE_SINGLE_OP_TASK_BUILD_TASK_UTILS_H_
