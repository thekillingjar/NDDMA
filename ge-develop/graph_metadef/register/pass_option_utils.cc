/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/pass_option_utils.h"
#include "register/optimization_option_registry.h"
#include "ge_common/debug/ge_log.h"
#include "common/checker.h"
#include "graph/ge_local_context.h"

namespace ge {
graphStatus PassOptionUtils::CheckIsPassEnabled(const std::string &pass_name, bool &is_enabled) {
  // 查看是否通过optimization_swtich配置过pass开关，配置过的话working_opt_names_to_value_中有pass_name这个option，且值为"on"或"off"
  const auto res = CheckIsPassEnabledByOption(pass_name, is_enabled);
  if (res == GRAPH_SUCCESS) {
    return GRAPH_SUCCESS;
  }
  GELOGD("The pass [%s] is not enabled by graph options", pass_name.c_str());
  /**
   * 原有逻辑分几种情况：
   * 1.PassOptionRegistry::pass_names_to_options_中没有pass关联的option，即pass未被注册，返回非零错误码
   * 2.PassOptionRegistry::pass_names_to_options_中有pass关联的option，但是其关联的option都没有注册，返回非零错误码
   * 3.PassOptionRegistry::pass_names_to_options_中有pass关联的option且有关联的option注册，根据passname到其关联的option中获取开关选项值，
   *  1) 若为空或者"true"，则is_enabled返回true（需要注册关联的option在注册时设置了该级别的默认值）
   *  2) 若为"false"，则is_enabled返回false
   */
  std::vector<std::string> opt_names;
  const auto ret = PassOptionRegistry::GetInstance().FindOptionNamesByPassName(pass_name, opt_names);
  if (ret != SUCCESS) {
    // 若Pass未被注册，返回非零错误码，由调用方判断如何处理
    GELOGI("The pass [%s] is not registered", pass_name.c_str());
    return ret;
  }

  // 当前最多支持两级开关，opt_names.size() <= 2
  std::vector<const OoInfo *> opt_infos;
  for (const auto &opt_name : opt_names) {
    const auto info_ptr = OptionRegistry::GetInstance().FindOptInfo(opt_name);
    if (info_ptr == nullptr) {
      GELOGW("Option [%s] of pass [%s] is not registered", opt_name.c_str(), pass_name.c_str());
      continue;
    }
    (void) opt_infos.emplace_back(info_ptr);
  }
  // Pass关联的选项均未注册,说明注册阶段遗漏了选项
  if (opt_infos.empty()) {
    GELOGW("the pass [%s] has no registered option", pass_name.c_str());
    return GRAPH_FAILED;
  }

  is_enabled = false;
  const auto &oo = GetThreadLocalContext().GetOo();
  for (auto it = opt_infos.crbegin(); it != opt_infos.crend(); ++it) {
    const auto opt = *it;
    std::string opt_value;
    if (oo.GetValue(opt->name, opt_value) == GRAPH_SUCCESS) {
      if (opt_value.empty() || (opt_value == "true")) {
        GELOGD("the pass [%s] is enabled, option [%s] is [%s]", pass_name.c_str(), opt->name.c_str(),
               opt_value.c_str());
        is_enabled = true;
        return GRAPH_SUCCESS;
      } else {
        GELOGD("the pass [%s] is disabled, option [%s] is [%s]", pass_name.c_str(), opt->name.c_str(),
               opt_value.c_str());
        return GRAPH_SUCCESS;
      }
    }
  }
  // OoTable中没有配置该Pass的开关选项，说明不使能
  GELOGD("the pass [%s] is disabled, option is not in working option table", pass_name.c_str());
  return GRAPH_SUCCESS;
}

graphStatus PassOptionUtils::CheckIsPassEnabledByOption(const std::string &pass_name, bool &is_enabled) {
  std::string opt_value;
  const auto res = GetThreadLocalContext().GetOo().GetValue(pass_name, opt_value);
  if (res != SUCCESS) {
    return GRAPH_FAILED;
  }

  if (opt_value == "on" || opt_value == "off") {
    GELOGD("The pass [%s] is configured by graph options, switch is [%s]", pass_name.c_str(), opt_value.c_str());
    is_enabled = (opt_value == "on");
    return GRAPH_SUCCESS;
  }

  return GRAPH_FAILED;
}
}  // namespace ge