/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_EXECUTOR_OPTION_H
#define AIR_CXX_EXECUTOR_OPTION_H

namespace gert {
enum class ExecutorType {
  // 顺序优先级执行器，基于优先级完成拓扑排序后，使用基于拓扑排序后的结果执行，
  // 本执行器调度速度最快，但是不支持条件/循环节点
  kSequentialPriority,

  // 基于拓扑的执行器，执行时基于拓扑动态计算ready节点，并做调度执行
  kTopological,

  // 基于拓扑的优先级执行器，在`kTopological`的基础上，将ready节点做优先级排序，总是优先执行优先级高的节点
  kTopologicalPriority,

  // 基于拓扑的多线程执行器
  kTopologicalMultiThread,

  kEnd
};

class VISIBILITY_EXPORT ExecutorOption {
 public:
  ExecutorOption() : executor_type_(ExecutorType::kEnd) {}
  explicit ExecutorOption(ExecutorType executor_type) : executor_type_(executor_type) {}
  ExecutorType GetExecutorType() const {
    return executor_type_;
  }
  virtual ~ExecutorOption() = default;

 private:
  ExecutorType executor_type_;
};
}  // namespace gert

#endif  // AIR_CXX_EXECUTOR_OPTION_H
