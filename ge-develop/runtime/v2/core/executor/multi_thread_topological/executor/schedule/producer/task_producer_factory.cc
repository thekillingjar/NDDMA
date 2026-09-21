/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "task_producer_factory.h"
#include "core/executor/multi_thread_topological/executor/schedule/producer/producers/chain_task_producer.h"
#include "core/executor/multi_thread_topological/executor/schedule/producer/producers/kernel_task_producer.h"
#include "core/executor/multi_thread_topological/executor/schedule/producer/producers/op_task_producer.h"
#include "core/executor/multi_thread_topological/executor/schedule/producer/producers/single_task_producer.h"
#include "task_producer_type.h"
#include "core/executor/multi_thread_topological/executor/schedule/config/task_producer_config.h"

namespace gert {
TaskProducer *TaskProducerFactory::Create(const TaskProducerConfig &cfg) const {
  TaskProducer *producer = nullptr;
  switch (cfg.type) {
    case TaskProducerType::CHAIN:
      producer = new (std::nothrow) ChainTaskProducer(cfg.prepared_task_count);
      break;

    case TaskProducerType::KERNEL:
      producer = new (std::nothrow) KernelTaskProducer(cfg.prepared_task_count, cfg.thread_num);
      break;

    case TaskProducerType::OP:
      producer = new (std::nothrow) OpTaskProducer(cfg.prepared_task_count);
      break;

    case TaskProducerType::SINGLE:
      producer = new (std::nothrow) SingleTaskProducer();
      break;

    default:
      break;
  }
  return producer;
}

void TaskProducerFactory::SetProducerType(TaskProducerType type) {
  producer_type_ = type;
  return;
}

const TaskProducerType &TaskProducerFactory::GetProducerType() const {
  return producer_type_;
}
}  // namespace gert
