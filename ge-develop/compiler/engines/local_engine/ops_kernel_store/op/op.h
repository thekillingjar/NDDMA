/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GE_LOCAL_ENGINE_OPS_KERNEL_STORE_OP_OP_H_
#define GE_GE_LOCAL_ENGINE_OPS_KERNEL_STORE_OP_OP_H_

#include <string>
#include "framework/common/ge_inner_error_codes.h"
#include "graph/node.h"

namespace ge {
struct RunContext;
namespace ge_local {
/**
 * The base class for all op.
 */
class GE_FUNC_VISIBILITY Op {
 public:
  Op(const Node &node, const RunContext &run_context);

  virtual ~Op() = default;

  virtual Status Run() = 0;

 protected:
  const RunContext &run_context_;
  const Node &node_;
  std::string name_;
  std::string type_;
};
}  // namespace ge_local
}  // namespace ge

#endif  // GE_GE_LOCAL_ENGINE_OPS_KERNEL_STORE_OP_OP_H_
