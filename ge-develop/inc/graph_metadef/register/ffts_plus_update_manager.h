/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_REGISTER_FFTS_PLUS_UPDATE_MANAGER_H_
#define INC_REGISTER_FFTS_PLUS_UPDATE_MANAGER_H_

#include <map>
#include <string>
#include <memory>
#include <functional>
#include <mutex>

#include "register/ffts_plus_task_update.h"

namespace ge {
using FftsCtxUpdatePtr = std::shared_ptr<FFTSPlusTaskUpdate>;
using FftsCtxUpdateCreatorFun = std::function<FftsCtxUpdatePtr()>;
class PluginManager;

class FftsPlusUpdateManager {
 public:
  static FftsPlusUpdateManager &Instance();
  /**
   * For load so to register FFTSPlusTaskUpdate subclass constructor.
   */
  Status Initialize();

  /**
   * Get FFTS Plus context by core type.
   * @param core_type: core type of Node
   * @return FFTS Plus Update instance.
   */
  FftsCtxUpdatePtr GetUpdater(const std::string &core_type) const;

  class FftsPlusUpdateRegistrar {
   public:
    FftsPlusUpdateRegistrar(const std::string &core_type, const FftsCtxUpdateCreatorFun &creator) {
      FftsPlusUpdateManager::Instance().RegisterCreator(core_type, creator);
    }
    ~FftsPlusUpdateRegistrar() = default;
  };

 private:
  FftsPlusUpdateManager() = default;
  ~FftsPlusUpdateManager();

  /**
   * Register FFTS Plus context update executor.
   * @param core_type: core type of Node
   * @param creator: FFTS Plus Update instance Creator.
   */
  void RegisterCreator(const std::string &core_type, const FftsCtxUpdateCreatorFun &creator);

  std::map<std::string, FftsCtxUpdateCreatorFun> creators_;
  std::unique_ptr<PluginManager> plugin_manager_;
};
} // namespace ge

#define REGISTER_FFTS_PLUS_CTX_UPDATER(core_type, task_clazz)             \
    REGISTER_FFTS_PLUS_CTX_TASK_UPDATER_UNIQ_HELPER(__COUNTER__, core_type, task_clazz)

#define REGISTER_FFTS_PLUS_CTX_TASK_UPDATER_UNIQ_HELPER(ctr, type, clazz) \
    REGISTER_FFTS_PLUS_CTX_TASK_UPDATER_UNIQ(ctr, type, clazz)

#define REGISTER_FFTS_PLUS_CTX_TASK_UPDATER_UNIQ(ctr, type, clazz)                              \
    ge::FftsPlusUpdateManager::FftsPlusUpdateRegistrar g_##type##_creator##ctr((type), []() {   \
      return std::shared_ptr<clazz>(new(std::nothrow) (clazz)());                             \
    })

#endif // INC_REGISTER_FFTS_PLUS_UPDATE_MANAGER_H_
