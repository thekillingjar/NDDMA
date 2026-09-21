/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_REGISTER_OP_EXTRA_GENTASK_REGISTRY_H
#define INC_REGISTER_OP_EXTRA_GENTASK_REGISTRY_H
#include <string>
#include <functional>
#include <vector>
#include "graph/node.h"
#include "proto/task.pb.h"
#include "external/ge_common/ge_api_types.h"
#include "common/opskernel/ops_kernel_info_types.h"
namespace fe {
using OpExtGenTaskFunc = ge::Status (*)(const ge::Node &node,
                                        ge::RunContext &context, std::vector<domi::TaskDef> &tasks);
using SKExtGenTaskFunc = ge::Status (*)(
  const ge::Node &node, std::vector<std::vector<domi::TaskDef>> &subTasks,
  const std::vector<ge::Node *> &sub_nodes, std::vector<domi::TaskDef> &tasks);

enum class ExtTaskType {
  kFftsPlusTask,
  kAicoreTask
};

class OpExtGenTaskRegistry {
 public:
  OpExtGenTaskRegistry() {};
  ~OpExtGenTaskRegistry() {};
  static OpExtGenTaskRegistry &GetInstance();
  OpExtGenTaskFunc FindRegisterFunc(const std::string &op_type) const;
  void Register(const std::string &op_type, OpExtGenTaskFunc const func);
  SKExtGenTaskFunc FindSKRegisterFunc(const std::string &op_type) const;
  void RegisterSKFunc(const std::string &op_type, SKExtGenTaskFunc const func);
  ExtTaskType GetExtTaskType(const std::string &op_type) const;
  void RegisterAicoreExtTask(const std::string &op_type);

 private:
  std::unordered_map<std::string, OpExtGenTaskFunc> names_to_register_func_;
  std::unordered_map<std::string, SKExtGenTaskFunc> types_to_sk_register_func_;
  std::unordered_set<std::string> aicore_ext_task_ops_;
};

class OpExtGenTaskRegister {
public:
    OpExtGenTaskRegister(const char *op_type, OpExtGenTaskFunc func) noexcept;
};

class SKExtGenTaskRegister {
 public:
     SKExtGenTaskRegister(const char *op_type, SKExtGenTaskFunc func) noexcept;
};

class ExtTaskTypeRegister {
 public:
  ExtTaskTypeRegister(const char *op_type, ExtTaskType type) noexcept;
};
}  // namespace fe

#ifdef __GNUC__
#define ATTRIBUTE_USED __attribute__((used))
#else
#define ATTRIBUTE_USED
#endif

#define REGISTER_NODE_EXT_GENTASK_COUNTER2(type, func, counter)                  \
  static const fe::OpExtGenTaskRegister g_reg_op_ext_gentask_##counter ATTRIBUTE_USED =  \
      fe::OpExtGenTaskRegister(type, func)
#define REGISTER_NODE_EXT_GENTASK_COUNTER(type, func, counter)                    \
  REGISTER_NODE_EXT_GENTASK_COUNTER2(type, func, counter)
#define REGISTER_NODE_EXT_GENTASK(type, func)                                \
  REGISTER_NODE_EXT_GENTASK_COUNTER(type, func, __COUNTER__)

#define REGISTER_SK_EXT_GENTASK_COUNTER2(type, func, counter)                  \
  static const fe::SKExtGenTaskRegister g_reg_op_ext_gentask_##counter ATTRIBUTE_USED =  \
      fe::SKExtGenTaskRegister(type, func)
#define REGISTER_SK_EXT_GENTASK_COUNTER(type, func, counter)                    \
  REGISTER_SK_EXT_GENTASK_COUNTER2(type, func, counter)
#define REGISTER_SK_EXT_GENTASK(type, func)                                \
  REGISTER_SK_EXT_GENTASK_COUNTER(type, func, __COUNTER__)

#define REGISTER_EXT_TASK_TYPE_COUNTER2(type, task_type, counter)                  \
  static const fe::ExtTaskTypeRegister g_reg_op_ext_gentask_##counter ATTRIBUTE_USED =  \
      fe::ExtTaskTypeRegister(#type, task_type)
#define REGISTER_EXT_TASK_TYPE_COUNTER(type, task_type, counter)                    \
  REGISTER_EXT_TASK_TYPE_COUNTER2(type, task_type, counter)
#define REGISTER_EXT_TASK_TYPE(type, task_type)                                \
  REGISTER_EXT_TASK_TYPE_COUNTER(type, task_type, __COUNTER__)
#endif // INC_REGISTER_OP_EXTRA_GENTASK_REGISTRY_H
