/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FLOW_FUNC_META_MULTI_FUNC_H
#define FLOW_FUNC_META_MULTI_FUNC_H

#include <functional>
#include <map>
#include "flow_func_defines.h"
#include "meta_run_context.h"
#include "meta_params.h"
#include "flow_msg.h"

namespace FlowFunc {
class FLOW_FUNC_VISIBILITY MetaMultiFunc {
 public:
  MetaMultiFunc() = default;

  ~MetaMultiFunc() = default;

  /**
   * @brief multi func init.
   * @return 0:success, other: failed.
   */
  virtual int32_t Init(const std::shared_ptr<MetaParams> &params) {
    (void)params;
    return FLOW_FUNC_SUCCESS;
  }
};

using PROC_FUNC_WITH_CONTEXT =
    std::function<int32_t(const std::shared_ptr<MetaRunContext> &, const std::vector<std::shared_ptr<FlowMsg>> &)>;

using MULTI_FUNC_CREATOR_FUNC = std::function<int32_t(std::shared_ptr<MetaMultiFunc> &multi_func,
                                                      std::map<AscendString, PROC_FUNC_WITH_CONTEXT> &proc_func_list)>;

/**
 * @brief register multi func creator.
 * @param flow_func_name cannot be null and must end with '\0'
 * @param func_creator multi func creator
 * @return if register success, true:success.
 */
FLOW_FUNC_VISIBILITY bool RegisterMultiFunc(const char *flow_func_name,
                                            const MULTI_FUNC_CREATOR_FUNC &func_creator) noexcept;

template <typename T>
class FlowFuncRegistrar {
 public:
  using CUSTOM_PROC_FUNC = std::function<int32_t(T *, const std::shared_ptr<MetaRunContext> &,
                                                 const std::vector<std::shared_ptr<FlowMsg>> &)>;

  FlowFuncRegistrar &RegProcFunc(const char *flow_func_name, const CUSTOM_PROC_FUNC &func) {
    using namespace std::placeholders;
    func_map_[flow_func_name] = func;
    (void)RegisterMultiFunc(flow_func_name, std::bind(&FlowFuncRegistrar::CreateMultiFunc, this, _1, _2));
    return *this;
  }

  int32_t CreateMultiFunc(std::shared_ptr<MetaMultiFunc> &multi_func,
                          std::map<AscendString, PROC_FUNC_WITH_CONTEXT> &proc_func_map) const {
    using namespace std::placeholders;
    T *flow_func_ptr = new (std::nothrow) T();
    if (flow_func_ptr == nullptr) {
      return FLOW_FUNC_FAILED;
    }
    multi_func.reset(flow_func_ptr);
    for (const auto &func : func_map_) {
      proc_func_map[func.first] = std::bind(func.second, flow_func_ptr, _1, _2);
    }
    return FLOW_FUNC_SUCCESS;
  }

 private:
  std::map<AscendString, CUSTOM_PROC_FUNC> func_map_;
};
/**
 * @brief define flow func REGISTRAR.
 * example:
 * FLOW_FUNC_REGISTRAR(UserFlowFunc).RegProcFunc("xxx_func", &UserFlowFunc::Proc1).
 *         RegProcFunc("xxx_func", &UserFlowFunc::Proc2);
 */
#define FLOW_FUNC_REGISTRAR(clazz)                                        \
  static FlowFunc::FlowFuncRegistrar<clazz> g_##clazz##FlowFuncRegistrar; \
  static auto &g_##clazz##Registrar = g_##clazz##FlowFuncRegistrar
}  // namespace FlowFunc
#endif  // FLOW_FUNC_META_MULTI_FUNC_H
