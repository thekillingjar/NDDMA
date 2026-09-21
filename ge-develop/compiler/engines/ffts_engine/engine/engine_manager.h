/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FFTS_ENGINE_ENGINE_ENGINE_MANAGER_H_
#define FFTS_ENGINE_ENGINE_ENGINE_MANAGER_H_

#include <map>
#include <memory>
#include <string>
#include "platform/platform_info.h"
#include "common/optimizer/graph_optimizer.h"
#include "inc/ffts_error_codes.h"

namespace ffts {
using GraphOptimizerPtr = std::shared_ptr<ge::GraphOptimizer>;

class EngineManager {
public:
  static EngineManager &Instance(const std::string &engine_name);

  Status Initialize(const std::map<string, string> &options, const std::string &engine_name, std::string &soc_version);

  Status Finalize();

  /*
   * to get the information of Graph Optimizer
   * param[out] the map of Graph Optimizer
   */
  void GetGraphOptimizerObjs(const map<string, GraphOptimizerPtr> &graph_optimizers,
                             const std::string &engine_name) const;

  std::string GetSocVersion();

private:
  EngineManager();
  ~EngineManager();
  std::string soc_version_;
  bool inited_;
};
}  // namespace ffts

#endif  // FFTS_ENGINE_ENGINE_ENGINE_MANAGER_H_
