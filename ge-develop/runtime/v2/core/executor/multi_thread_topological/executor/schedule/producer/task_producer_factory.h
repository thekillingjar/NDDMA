/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_TASK_PRODUCER_FACTORY_H
#define AIR_CXX_RUNTIME_V2_TASK_PRODUCER_FACTORY_H

#include "task_producer.h"
#include "core/executor/multi_thread_topological/executor/schedule/config/task_producer_config.h"

namespace gert {
class TaskProducerFactory {
 public:
  static TaskProducerFactory &GetInstance() {
    static TaskProducerFactory instance;
    return instance;
  }

  TaskProducerFactory(const TaskProducerFactory &) = delete;
  TaskProducerFactory &operator=(const TaskProducerFactory &) = delete;

  TaskProducer *Create(const TaskProducerConfig &cfg) const;
  void SetProducerType(TaskProducerType type);
  const TaskProducerType &GetProducerType() const;

 private:
  TaskProducerFactory() noexcept = default;
  TaskProducerType producer_type_{kTaskProducerTypeDefault};
};
}  // namespace gert
#endif  // AIR_CXX_RUNTIME_V2_TASK_PRODUCER_FACTORY_H
