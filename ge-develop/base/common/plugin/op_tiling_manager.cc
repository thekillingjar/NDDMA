/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/plugin/op_tiling_manager.h"

#include <string>
#include <array>

#include "graph_metadef/common/plugin/plugin_manager.h"
#include "common/compile_profiling/ge_call_wrapper.h"
#include "framework/common/debug/log.h"
#include "mmpa/mmpa_api.h"
#include "base/err_msg.h"

namespace ge {
void OpTilingManager::ClearHandles() noexcept {
  for (const auto &handle : handles_) {
    if (mmDlclose(handle.second) != 0) {
      const char_t *error = mmDlerror();
      GE_IF_BOOL_EXEC(error == nullptr, error = "");
      GELOGE(FAILED, "[Close][Handle]Failed, handle of %s: %s", handle.first.c_str(), error);
      REPORT_INNER_ERR_MSG("E19999", "Failed to close handle of %s: %s",
                        handle.first.c_str(), error);
    }
  }
  handles_.clear();
}

OpTilingManager::~OpTilingManager() { ClearHandles(); }

void OpTilingManager::LoadSo() {
  FuncPerfScope func_perf_scope("OpTilingManager", __FUNCTION__);
  std::string op_tiling_path;
  const Status ret = PluginManager::GetOpTilingPath(op_tiling_path);
  if (ret != SUCCESS) {
    GELOGW("Failed to get op tiling path!");
    return;
  }

  std::string os_type;
  std::string cpu_type;
  PluginManager::GetCurEnvPackageOsAndCpuType(os_type, cpu_type);

  std::vector<std::string> path_vec;
  PluginManager::SplitPath(op_tiling_path, path_vec);
  for (const auto &path : path_vec) {
    std::string root_path = path + "op_master/";
    std::string so_name = "libopmaster.so";
    std::array<char_t, MMPA_MAX_PATH> resolved_path = {};
    const INT32 result = mmRealPath(root_path.c_str(), &(resolved_path[0U]), MMPA_MAX_PATH);
    if (result != EN_OK) {
      GELOGW("[FindSo][Check] Get path with op_master [%s] failed", root_path.c_str());
      root_path = path + "op_tiling/";
      so_name = "liboptiling.so";
    }
    std::string so_path = root_path + "lib/" + os_type + "/" + cpu_type + "/" + so_name;
    void *handle = mmDlopen(so_path.c_str(),
                            static_cast<int32_t>(static_cast<uint32_t>(MMPA_RTLD_NOW) |
                            static_cast<uint32_t>(MMPA_RTLD_GLOBAL)));
    if (handle == nullptr) {
      GELOGW("Failed to dlopen %s! errmsg:%s", so_path.c_str(), mmDlerror());
      so_path = root_path + so_name;
      handle = mmDlopen(so_path.c_str(),
                        static_cast<int32_t>(static_cast<uint32_t>(MMPA_RTLD_NOW) |
                        static_cast<uint32_t>(MMPA_RTLD_GLOBAL)));
      if (handle == nullptr) {
        GELOGW("Failed to dlopen %s! errmsg:%s", so_path.c_str(), mmDlerror());
      } else {
        GELOGI("dlopen file %s successfully!", so_path.c_str());
        handles_[so_path] = handle;
      }
    } else {
      GELOGI("dlopen file %s successfully!", so_path.c_str());
      handles_[so_path] = handle;
    }
  }
}

OpTilingManager &OpTilingManager::GetInstance() {
  static OpTilingManager instance;
  return instance;
}
}  // namespace ge
