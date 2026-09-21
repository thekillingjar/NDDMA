/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_OPTIMIZER_ADAPTER_COMMON_TASK_BUILDER_ADAPTER_FACTORY_H_
#define AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_OPTIMIZER_ADAPTER_COMMON_TASK_BUILDER_ADAPTER_FACTORY_H_

#include "common/aicore_util_types.h"
#include "adapter/adapter_itf/task_builder_adapter.h"
#include "graph/node.h"

namespace fe {
class TaskBuilderAdapterFactory {
 public:
  TaskBuilderAdapterFactory(const TaskBuilderAdapterFactory &) = delete;
  TaskBuilderAdapterFactory &operator=(const TaskBuilderAdapterFactory &) = delete;

  /**
   * Get the singleton Instance of TaskBuilderAdapterFactory
   * @return Instance
   */
  static TaskBuilderAdapterFactory &Instance();

  /**
   * Create task builder adapter pointer
   * @param op_impl_type
   * @return
   */
  TaskBuilderAdapterPtr CreateTaskBuilderAdapter(const OpImplType &op_impl_type, const ge::Node &node,
                                                 TaskBuilderContext &context) const;

 private:
  TaskBuilderAdapterFactory();
  ~TaskBuilderAdapterFactory();
  static const std::vector<OpImplType> TBE_IMPL_TYPE_VEC;
};
}  // namespace fe
#endif  // AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_OPTIMIZER_ADAPTER_COMMON_TASK_BUILDER_ADAPTER_FACTORY_H_
