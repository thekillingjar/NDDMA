/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BASE_COMMON_PROCESS_NODE_ENGINE_MANAGER_H_
#define BASE_COMMON_PROCESS_NODE_ENGINE_MANAGER_H_

#include <map>
#include <memory>
#include <string>
#include <vector>
#include <mutex>
#include <atomic>

#include "graph_metadef/common/plugin/plugin_manager.h"
#include "framework/common/ge_inner_error_codes.h"
#include "process_node_engine.h"

namespace ge {
using CreateFn = ProcessNodeEngine *(*)();

class ProcessNodeEngineManager {
 public:
  ProcessNodeEngineManager(const ProcessNodeEngineManager &other) = delete;
  ProcessNodeEngineManager &operator=(const ProcessNodeEngineManager &other) & = delete;
  static ProcessNodeEngineManager &GetInstance();
  Status Initialize(const std::map<std::string, std::string> &options);
  Status Finalize();
  Status RegisterEngine(const std::string &engine_id, const ProcessNodeEnginePtr &engine, CreateFn const fn);
  ProcessNodeEnginePtr GetEngine(const std::string &engine_id) const;
  ProcessNodeEnginePtr CloneEngine(const std::string &engine_id) const;
  inline const std::map<std::string, ProcessNodeEnginePtr> &GetEngines() const { return engines_map_; }
  bool IsEngineRegistered(const std::string &engine_id) const;

 private:
  ProcessNodeEngineManager() = default;
  ~ProcessNodeEngineManager() = default;

 private:
  PluginManager plugin_mgr_;
  std::map<std::string, ProcessNodeEnginePtr> engines_map_;
  std::map<std::string, CreateFn> engines_create_map_;
  std::atomic<bool> init_flag_{false};
  mutable std::mutex mutex_;
};

class ProcessNodeEngineRegisterar {
public:
  ProcessNodeEngineRegisterar(const std::string &engine_id, CreateFn const fn) noexcept;
  ~ProcessNodeEngineRegisterar() = default;
  ProcessNodeEngineRegisterar(const ProcessNodeEngineRegisterar &other) = delete;
  ProcessNodeEngineRegisterar &operator=(const ProcessNodeEngineRegisterar &other) & = delete;
};
}  // namespace ge

#define REGISTER_PROCESS_NODE_ENGINE(id, engine)                                                        \
  static ge::ProcessNodeEngineRegisterar g_##engine##_register __attribute__((unused))((id),            \
      []()->::ge::ProcessNodeEngine * { return new (std::nothrow) ge::engine(); })

#endif  // BASE_COMMON_PROCESS_NODE_ENGINE_MANAGER_H_