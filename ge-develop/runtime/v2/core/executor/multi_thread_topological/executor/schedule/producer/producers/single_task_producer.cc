/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "single_task_producer.h"
#include "common/checker.h"
#include "core/execution_data.h"

namespace gert {
SingleTaskProducer::SingleTaskProducer() {}

ge::Status SingleTaskProducer::StartUp() {
  auto execution_data = reinterpret_cast<const ExecutionData *>(execution_data_);
  for (size_t i = 0U; i < execution_data->start_num; ++i) {
    TravelNodes(execution_data->start_nodes[i]);
  }
  tasks_.push_back(execTask_);
  return ge::SUCCESS;
}

ge::Status SingleTaskProducer::Prepare(const void *execution_data) {
  GE_ASSERT_TRUE(execution_data != nullptr);
  execution_data_ = execution_data;
  return ge::SUCCESS;
}

void SingleTaskProducer::TravelNodes(Node *node) {
  execTask_.AddKernel(node);
  auto execution_data = reinterpret_cast<const ExecutionData *>(execution_data_);
  Watcher *watchers = execution_data->node_watchers[node->node_id];
  for (size_t i = 0U; i < watchers->watch_num; ++i) {
    NodeIdentity node_id = watchers->node_ids[i];
    if (--execution_data->node_indegrees[node_id] == 0) {
      TravelNodes(execution_data->base_ed.nodes[node_id]);
      execution_data->node_indegrees[node_id] = execution_data->node_indegrees_backup[node_id];
    }
  }
}

ge::Status SingleTaskProducer::DoRecycleTask(ExecTask &) {
  return ge::SUCCESS;
}

void SingleTaskProducer::Dump() const {
  GEEVENT("|-- Producer [type : single]");
}
}  // namespace gert
