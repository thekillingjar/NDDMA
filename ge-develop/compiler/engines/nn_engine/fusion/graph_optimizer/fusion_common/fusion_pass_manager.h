/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSION_ENGINE_FUSION_GRAPH_OPTIMIZER_FUSION_COMMON_FUSION_PASS_MANAGER_H_
#define FUSION_ENGINE_FUSION_GRAPH_OPTIMIZER_FUSION_COMMON_FUSION_PASS_MANAGER_H_

#include <map>
#include <memory>
#include <string>
#include <vector>
#include "common/aicore_util_types.h"
#include "common/plugin_manager.h"
#include "graph/compute_graph.h"

namespace fe {
class FusionPassManager {
 public:
  FusionPassManager();
  ~FusionPassManager();

  Status Initialize(const std::string &engine_name);
  Status Finalize();

 private:
  Status InitFusionPassPlugin(const std::string &engine_name);
  Status OpenPassPath(const std::string &file_path, std::vector<string> &get_pass_files);
  void CloseFusionPassPlugin();

 private:
  bool init_flag_;
  vector<PluginManagerPtr> fusion_pass_plugin_manager_vec_;
};
}  // namespace fe
#endif  // FUSION_ENGINE_FUSION_GRAPH_OPTIMIZER_FUSION_COMMON_FUSION_PASS_MANAGER_H_
