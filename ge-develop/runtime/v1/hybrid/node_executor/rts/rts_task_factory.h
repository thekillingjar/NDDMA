/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_HYBRID_NODE_EXECUTOR_RTS_TASK_FACTORY_H_
#define GE_HYBRID_NODE_EXECUTOR_RTS_TASK_FACTORY_H_

#include "hybrid/node_executor/rts/rts_node_task.h"

namespace ge {
namespace hybrid {
using RtsNodeTaskPtr = std::shared_ptr<RtsNodeTask>;
using RtsTaskCreatorFun = std::function<RtsNodeTaskPtr()>;

class RtsTaskFactory {
 public:
  static RtsTaskFactory &GetInstance() {
    static RtsTaskFactory instance;
    return instance;
  }

  RtsNodeTaskPtr Create(const std::string &task_type) const;

  class RtsTaskRegistrar {
   public:
    RtsTaskRegistrar(const std::string &task_type, const RtsTaskCreatorFun &creator) noexcept {
      RtsTaskFactory::GetInstance().RegisterCreator(task_type, creator);
    }
    ~RtsTaskRegistrar() = default;
  };

 private:
  RtsTaskFactory() = default;
  ~RtsTaskFactory() = default;

  /**
   * Register build of executor
   * @param executor_type   type of executor
   * @param builder         build function
   */
  void RegisterCreator(const std::string &task_type, const RtsTaskCreatorFun &creator);

  std::map<std::string, RtsTaskCreatorFun> creators_;
};
}  // namespace hybrid
}  // namespace ge

#define REGISTER_RTS_TASK_CREATOR(task_type, task_clazz) \
    REGISTER_RTS_TASK_CREATOR_UNIQ_HELPER(__COUNTER__, task_type, task_clazz)

#define REGISTER_RTS_TASK_CREATOR_UNIQ_HELPER(ctr, type, clazz) \
  static RtsTaskFactory::RtsTaskRegistrar g_##type##_Creator##ctr(type, \
                                                                  []()->RtsNodeTaskPtr { return MakeShared<clazz>(); })

#endif // GE_HYBRID_NODE_EXECUTOR_RTS_TASK_FACTORY_H_
