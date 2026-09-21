/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GE_LOCAL_ENGINE_OPS_KERNEL_STORE_OP_NO_OP_H_
#define GE_GE_LOCAL_ENGINE_OPS_KERNEL_STORE_OP_NO_OP_H_

#include "engines/local_engine/ops_kernel_store/op/op.h"

namespace ge {
namespace ge_local {
class GE_FUNC_VISIBILITY NoOp : public Op {
 public:
  NoOp(const Node &node, RunContext &run_context);

  ~NoOp() override = default;

  NoOp &operator=(const NoOp &op) = delete;

  NoOp(const NoOp &op) = delete;

  /**
   *  @brief generate task.
   *  @return result
   */
  ge::Status Run() override;
};
}  // namespace ge_local
}  // namespace ge

#endif  // GE_GE_LOCAL_ENGINE_OPS_KERNEL_STORE_OP_NO_OP_H_
