/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef METADEF_CXX_INC_GRAPH_BASE_CUSTOM_OP_H
#define METADEF_CXX_INC_GRAPH_BASE_CUSTOM_OP_H
#include "exe_graph/runtime/eager_op_execution_context.h"

namespace ge {
class BaseCustomOp {
 public:
  virtual graphStatus Execute(gert::EagerOpExecutionContext *ctx) = 0;
  virtual ~BaseCustomOp() = default;
};

class EagerExecuteOp : public BaseCustomOp {
 public:
  /**
   * Eager类自定义OP的执行函数
   * @param ctx。执行时上下文，可通过上下文获取input tensor，分配输出内存，分配workspace等
   * @return 状态码
   */
  virtual graphStatus Execute(gert::EagerOpExecutionContext *ctx) = 0;
};

using BaseOpCreator = std::function<std::unique_ptr<BaseCustomOp>()>;

class CustomOpCreatorRegister {
public:
  CustomOpCreatorRegister(const AscendString &operator_type, const BaseOpCreator &op_creator);
  ~CustomOpCreatorRegister() = default;
};
}  // namespace ge

#define REG_JOIN(g_register, y) g_register##y
#define REG_AUTO_MAPPING_OP(custom_op_class) REG_AUTO_MAPPING_OP_UNIQ(__COUNTER__, custom_op_class)
#define REG_AUTO_MAPPING_OP_UNIQ(ctr, custom_op_class)             \
  static const ge::CustomOpCreatorRegister REG_JOIN(custom_op_register, ctr)( \
      #custom_op_class, []() -> std::unique_ptr<ge::BaseCustomOp> { return std::make_unique<custom_op_class>(); })

#endif  // METADEF_CXX_INC_GRAPH_BASE_CUSTOM_OP_H
