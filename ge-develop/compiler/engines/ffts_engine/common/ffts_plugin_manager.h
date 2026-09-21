/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FFTS_ENGINE_COMMON_PLUGIN_MANAGER_H_
#define FFTS_ENGINE_COMMON_PLUGIN_MANAGER_H_
#include <dlfcn.h>
#include <functional>
#include <iostream>
#include <string>
#include "graph_optimizer/graph_optimize_register_error_codes.h"
#include "inc/ffts_log.h"
#include "inc/ffts_error_codes.h"

namespace ffts {
using std::vector;
using std::string;
using std::function;

class PluginManager {
 public:
  explicit PluginManager(const string &name) : so_name(name) {}

  ~PluginManager() {}

  /*
  *  @ingroup ffts
  *  @brief   return tbe so name
  *  @return  tbe so name
  */
  const string GetSoName() const;

  /*
  *  @ingroup ffts
  *  @brief   initial resources needed by TbeCompilerAdapter,
  *  such as dlopen so files and load function symbols etc.
  *  @return  SUCCESS or FAILED
  */
  Status OpenPlugin(const string& path);

  /*
*  @ingroup ffts
*  @brief   release resources needed by TbeCompilerAdapter,
*  such as dlclose so files and free tbe resources etc.
*  @return  SUCCESS or FAILED
*/
  Status CloseHandle();

  /*
  *  @ingroup ffts
  *  @brief   load function symbols from so files
  *  @return  SUCCESS or FAILED
  */
  template <typename R, typename... Types>
  Status GetFunction(const string& func_name, function<R(Types... args)>& func) const {
    func = (R(*)(Types...))dlsym(handle, func_name.c_str());

    FFTS_CHECK(func == nullptr, FFTS_LOGW("Failed to get function %s in %s!", func_name.c_str(), so_name.c_str()),
             return FAILED);

    return SUCCESS;
  }

 private:
  string so_name;
  void* handle{nullptr};
};
}  // namespace ffts

#endif  // FFTS_ENGINE_COMMON_PLUGIN_MANAGER_H_
