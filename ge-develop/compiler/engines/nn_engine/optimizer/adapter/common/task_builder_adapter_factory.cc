/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "adapter/common/task_builder_adapter_factory.h"
#include "common/fe_log.h"
#include "adapter/tbe_adapter/tbe_task_builder_adapter.h"
#include "common/fe_log.h"

namespace fe {
const std::vector<OpImplType> TaskBuilderAdapterFactory::TBE_IMPL_TYPE_VEC = {
    EN_IMPL_HW_TBE, EN_IMPL_CUSTOM_TBE, EN_IMPL_PLUGIN_TBE, EN_IMPL_VECTOR_CORE_HW_TBE,
    EN_IMPL_VECTOR_CORE_CUSTOM_TBE, EN_IMPL_NON_PERSISTENT_CUSTOM_TBE
};

TaskBuilderAdapterFactory &TaskBuilderAdapterFactory::Instance() {
  static TaskBuilderAdapterFactory task_builder_adapter_factory;
  return task_builder_adapter_factory;
}

TaskBuilderAdapterPtr TaskBuilderAdapterFactory::CreateTaskBuilderAdapter(const OpImplType &op_impl_type,
                                                                          const ge::Node &node,
                                                                          TaskBuilderContext &context) const {
  TaskBuilderAdapterPtr task_builder_adapter_ptr = nullptr;
  if (std::find(TBE_IMPL_TYPE_VEC.begin(), TBE_IMPL_TYPE_VEC.end(), op_impl_type) != TBE_IMPL_TYPE_VEC.end()) {
    FE_MAKE_SHARED(task_builder_adapter_ptr = std::make_shared<TbeTaskBuilderAdapter>(node, context), return nullptr);
    if (task_builder_adapter_ptr == nullptr) {
      FE_LOGW("Failed to create TbeTaskBuilderAdapter pointer for node[%s].", node.GetName().c_str());
    }
  }
  return task_builder_adapter_ptr;
}

TaskBuilderAdapterFactory::TaskBuilderAdapterFactory() {}
TaskBuilderAdapterFactory::~TaskBuilderAdapterFactory() {}
}  // namespace fe
