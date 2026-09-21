/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_OPTIMIZER_ADAPTER_ADAPTER_ITF_TASK_BUILDER_ADAPTER_H_
#define AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_OPTIMIZER_ADAPTER_ADAPTER_ITF_TASK_BUILDER_ADAPTER_H_

#include <memory>
#include <string>
#include <vector>
#include "proto/task.pb.h"
#include "graph/node.h"
#include "register/graph_optimizer/graph_optimize_register_error_codes.h"

namespace fe {
extern const int64_t kMaxWeightOffset;
class TaskBuilderAdapter;
using TaskBuilderAdapterPtr = std::shared_ptr<TaskBuilderAdapter>;

struct TaskBuilderContext {
  uint64_t dataMemSize = 0;
  uint8_t *dataMemBase = nullptr;
  uint64_t weightMemSize = 0;
  uint8_t *weightMemBase = nullptr;
  ge::Buffer weightBufferHost;
  int32_t stream_id = 0;
  std::map<int64_t, uint8_t *> mem_type_to_data_mem_base;
  std::map<int64_t, uint64_t> mem_type_to_data_mem_size;
};

struct TaskArgs {
  vector<void *> input_addrs;
  vector<void *> output_addrs;
  // Workspace
  vector<void *> workspace_addrs;
  // ArgsInfo
  vector<domi::ArgsInfo> kernel_args_info;
};

class TaskBuilderAdapter {
 public:
  TaskBuilderAdapter(const ge::Node &node, TaskBuilderContext &context);
  virtual ~TaskBuilderAdapter();

  /*
   * @ingroup fe
   * @brief   Init TaskBuilderAdapter
   * @return  SUCCESS or FAILED
   */
  virtual Status Init();

  /*
   * @ingroup fe
   * @brief   Run TaskBuilderAdapter
   * @return  SUCCESS or FAILED
   */
  virtual Status Run(domi::TaskDef &task_def);

  ge::ConstOpDescPtr GetOpDesc() { return op_desc_; }

  void GetTaskArgs(TaskArgs &args_info) const;

  void FeedArgsInfo(const domi::ArgsInfo_ArgsType &type, const domi::ArgsInfo_ArgsFormat &arg_format,
                    const size_t &input_index);

 protected:
  virtual Status InitInput() = 0;
  virtual Status InitOutput();
  virtual Status InitWorkspace();
  virtual Status InitInputL1Addr();
  static Status CheckOffsetAndSize(int64_t offset, uint64_t space_size, uint64_t total_size);

 protected:
  const ge::Node &node_;
  ge::OpDescPtr op_desc_;
  TaskBuilderContext &context_;
  vector<domi::ArgsInfo> kernel_args_info_;

  vector<void *> input_addrs_;

  vector<void *> output_addrs_;

  // Workspace
  vector<void *> workspace_addrs_;
  vector<int64_t> workspace_sizes_;

  vector<void *> input_l1_addrs_;
  bool is_unknown_graph_;

 private:
  TaskBuilderAdapter(const TaskBuilderAdapter &) = delete;
  TaskBuilderAdapter &operator=(const TaskBuilderAdapter &) = delete;

  Status VerifyWeights();
};
}  // namespace fe
#endif  // AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_OPTIMIZER_ADAPTER_ADAPTER_ITF_TASK_BUILDER_ADAPTER_H_
