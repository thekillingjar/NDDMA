/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_OPSKERNEL_MANAGER_OPS_KERNEL_BUILDER_MANAGER_H_
#define GE_OPSKERNEL_MANAGER_OPS_KERNEL_BUILDER_MANAGER_H_

#include "graph_metadef/common/plugin/plugin_manager.h"
#include "common/ge_common/fmk_error_codes.h"
#include "common/opskernel/ops_kernel_builder.h"
#include "ge/ge_api_error_codes.h"
#include "register/ops_kernel_builder_registry.h"

namespace ge {
class GE_OBJECT_VISIBILITY OpsKernelBuilderManager {
 public:
  ~OpsKernelBuilderManager();

  static OpsKernelBuilderManager &Instance();

  // opsKernelManager initialize, load all opsKernelInfoStore and graph_optimizer
  Status Initialize(const std::map<std::string, std::string> &options,
                    const std::string &path_base, const bool is_train = true);

  // opsKernelManager finalize, unload all opsKernelInfoStore and graph_optimizer
  Status Finalize();

  // get opsKernelIBuilder by name
  OpsKernelBuilderPtr GetOpsKernelBuilder(const std::string &name) const;

  // get all opsKernelBuilders
  const std::map<std::string, OpsKernelBuilderPtr> &GetAllOpsKernelBuilders() const;

  Status CalcOpRunningParam(Node &node) const;

  Status GenerateTask(const Node &node, RunContext &context, std::vector<domi::TaskDef> &tasks,
                      const bool atomic_engine_flag = true) const;

  Status UpdateTask(const Node &node, std::vector<domi::TaskDef> &tasks) const;

 private:
  OpsKernelBuilderManager() = default;
  Status GetLibPaths(const std::map<std::string, std::string> &options,
                     const std::string &path_base, std::string &lib_paths) const;

  std::unique_ptr<PluginManager> plugin_manager_;
};
}  // namespace ge
#endif  // GE_OPSKERNEL_MANAGER_OPS_KERNEL_BUILDER_MANAGER_H_
