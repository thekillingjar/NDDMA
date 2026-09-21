/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_1CB1E40F5C1F4C78AC6CB75D6A8B4179
#define INC_1CB1E40F5C1F4C78AC6CB75D6A8B4179

#include "stdint.h"
#include "ge_running_env/fake_ns.h"
#include <string>
#include "graph/graph.h"

FAKE_NS_BEGIN

struct TinyModelPartitionTable;
struct ModelFileHeader;
struct ModelData;
struct TinyModelPartitionMemInfo;
struct Graph;

struct ModelDataBuilder {
  ModelDataBuilder(ModelData &);
  ModelDataBuilder& AddGraph(const Graph &graph);
  ModelDataBuilder& AddTask();
  ModelDataBuilder& AddTask(int kernel_type, int op_index, size_t arg_size = 64U);
  ModelDataBuilder& AddTask(int kernel_type, const char *node_name);
  ModelDataBuilder& AddAicpuTask(int op_index);
  void Build();

 private:
  void Init();

 private:
  ModelData &model;
  uint8_t *model_data;
  ModelFileHeader *model_header;
  TinyModelPartitionTable *partition_table;
  TinyModelPartitionMemInfo &graph_info_;
  TinyModelPartitionMemInfo &task_info_;
  uint32_t op_index_;
  uint64_t mem_offset;
  ge::Graph graph_;
};

FAKE_NS_END
#endif
