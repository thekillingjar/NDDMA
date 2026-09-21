/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSION_ENGINE_UTILS_COMMON_PLUGIN_MANAGER_H_
#define FUSION_ENGINE_UTILS_COMMON_PLUGIN_MANAGER_H_

#include <functional>
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include "graph_optimizer/graph_optimize_register_error_codes.h"
#include "common/fe_log.h"
#include "mmpa/mmpa_api.h"

namespace fe {
using std::vector;
using std::string;
using std::function;

class PluginManager {
 public:
  explicit PluginManager(const string &name) : so_name(name) {}

  ~PluginManager() {}

  /*
  *  @ingroup fe
  *  @brief   return tbe so name
  *  @return  tbe so name
  */
  const string GetSoName() const;

  /*
  *  @ingroup fe
  *  @brief   initial resources needed by TbeCompilerAdapter,
  *  such as dlopen so files and load function symbols etc.
  *  @return  SUCCESS or FAILED
  */
  Status OpenPlugin(const string& path);

  /*
*  @ingroup fe
*  @brief   release resources needed by TbeCompilerAdapter,
*  such as dlclose so files and free tbe resources etc.
*  @return  SUCCESS or FAILED
*/
  Status CloseHandle();

  /*
  *  @ingroup fe
  *  @brief   load function symbols from so files
  *  @return  SUCCESS or FAILED
  */
  template <typename R, typename... Types>
  Status GetFunction(const string& func_name, function<R(Types... args)>& func) const {
    func = (R(*)(Types...))mmDlsym(handle, func_name.c_str());

    FE_CHECK(func == nullptr, FE_LOGW("Failed to get function %s in %s!", func_name.c_str(), so_name.c_str()),
             return FAILED);

    return SUCCESS;
  }

  template <typename R, typename... Types>
  Status GetFunctionFromTbePlugin(const string& func_name, function<R(Types... args)>& func) {
    Status ret = GetFunction(func_name, func);
    if (ret != SUCCESS) {
      REPORT_FE_ERROR("[GraphOpt][InitTbeFunc]: Failed to dlsym function %s.", func_name.c_str());
      // No matter we close it successfully or not, we will return FAILED because we failed to get fucntion.
      (void)CloseHandle();
      return FAILED;
    }
    return SUCCESS;
  }

 private:
  string so_name;
  void* handle{nullptr};
};
using PluginManagerPtr = std::shared_ptr<fe::PluginManager>;
}  // namespace fe
#endif  // FUSION_ENGINE_UTILS_COMMON_PLUGIN_MANAGER_H_
