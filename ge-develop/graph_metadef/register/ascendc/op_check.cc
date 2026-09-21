/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/op_check_register.h"
#include "framework/common/debug/ge_log.h"

namespace optiling {
std::map<ge::AscendString, std::map<ge::AscendString, OP_CHECK_FUNC>>
    OpCheckFuncRegistry::check_op_capability_instance_;
std::map<ge::AscendString, GEN_SIMPLIFIEDKEY_FUNC> OpCheckFuncRegistry::gen_simplifiedkey_instance_;
std::map<ge::AscendString, PARAM_GENERALIZE_FUNC> OpCheckFuncRegistry::param_generalize_instance_;
std::map<ge::AscendString, std::map<ge::AscendString, REPLAY_FUNC>> OpCheckFuncRegistry::replay_instance_;

void OpCheckFuncRegistry::RegisterOpCapability(const ge::AscendString &check_type, const ge::AscendString &op_type,
                                               OP_CHECK_FUNC func) {
  check_op_capability_instance_[check_type][op_type] = func;
  GELOGI("RegisterOpCapability: check_type:%s, op_type:%s, funcPointer:%p, registered count:%zu",
         check_type.GetString(), op_type.GetString(), func, check_op_capability_instance_[check_type].size());
}

OP_CHECK_FUNC OpCheckFuncRegistry::GetOpCapability(const ge::AscendString &check_type,
                                                   const ge::AscendString &op_type) {
  const auto &check_map_it = check_op_capability_instance_.find(check_type);
  if (check_map_it == check_op_capability_instance_.end()) {
    GELOGW("GetOpCapability: check_type:%s, op_type:%s, cannot find check_type.", check_type.GetString(),
           op_type.GetString());
    return nullptr;
  }
  const auto &func_it = check_map_it->second.find(op_type);
  if (func_it == check_map_it->second.end()) {
    GELOGW("GetOpCapability: check_type:%s, op_type:%s, cannot find op_type.", check_type.GetString(),
           op_type.GetString());
    return nullptr;
  }
  return func_it->second;
}

void OpCheckFuncRegistry::RegisterGenSimplifiedKeyFunc(const ge::AscendString &op_type, GEN_SIMPLIFIEDKEY_FUNC func) {
  gen_simplifiedkey_instance_[op_type] = func;
  GELOGI("RegisterGenSimplifiedKeyFunc: op_type:%s, registered count:%zu", op_type.GetString(),
         gen_simplifiedkey_instance_.size());
}

GEN_SIMPLIFIEDKEY_FUNC OpCheckFuncRegistry::GetGenSimplifiedKeyFun(const ge::AscendString &op_type) {
  const auto &func_it = gen_simplifiedkey_instance_.find(op_type);
  if (func_it == gen_simplifiedkey_instance_.end()) {
    GELOGW("GetGenSimplifiedKeyFun: op_type:%s, cannot find func by op_type.", op_type.GetString());
    return nullptr;
  }
  return func_it->second;
}

PARAM_GENERALIZE_FUNC OpCheckFuncRegistry::GetParamGeneralize(const ge::AscendString &op_type) {
  const auto &func_it = param_generalize_instance_.find(op_type);
  if (func_it == param_generalize_instance_.end()) {
    GELOGW("GetParamGeneralize: op_type:%s, cannot find op_type.", op_type.GetString());
    return nullptr;
  }
  return func_it->second;
}

void OpCheckFuncRegistry::RegisterParamGeneralize(const ge::AscendString &op_type, PARAM_GENERALIZE_FUNC func) {
  param_generalize_instance_[op_type] = func;
  GELOGI("RegisterParamGeneralize: op_type:%s, funcPointer:%p, registered count:%zu", op_type.GetString(), func,
         param_generalize_instance_.size());
}

void OpCheckFuncRegistry::RegisterReplay(const ge::AscendString &op_type, const ge::AscendString &soc_version,
                                         REPLAY_FUNC func) {
  replay_instance_[op_type][soc_version] = func;
  GELOGI("RegisterReplay: op_type:%s, soc_version:%s funcPointer:%p, registered count:%zu", op_type.GetString(),
         soc_version.GetString(), func, replay_instance_[op_type].size());
}

REPLAY_FUNC OpCheckFuncRegistry::GetReplay(const ge::AscendString &op_type, const ge::AscendString &soc_version) {
  const auto &soc_map_it = replay_instance_.find(op_type);
  if (soc_map_it == replay_instance_.end()) {
    GELOGW("GetReplay: op_type:%s, soc_version:%s, cannot find op_type.", op_type.GetString(), soc_version.GetString());
    return nullptr;
  }
  const auto &func_it = soc_map_it->second.find(soc_version);
  if (func_it == soc_map_it->second.end()) {
    GELOGW("GetReplay: op_type:%s, soc_version:%s, cannot find soc_version.", op_type.GetString(),
           soc_version.GetString());
    return nullptr;
  }
  return func_it->second;
}

OpCheckFuncHelper::OpCheckFuncHelper(const ge::AscendString &check_type, const ge::AscendString &op_type,
                                     OP_CHECK_FUNC func) {
  OpCheckFuncRegistry::RegisterOpCapability(check_type, op_type, func);
}

OpCheckFuncHelper::OpCheckFuncHelper(const ge::AscendString &op_type, PARAM_GENERALIZE_FUNC func) {
  OpCheckFuncRegistry::RegisterParamGeneralize(op_type, func);
}

ReplayFuncHelper::ReplayFuncHelper(const ge::AscendString &op_type, const ge::AscendString &soc_version,
                                   REPLAY_FUNC func) {
  OpCheckFuncRegistry::RegisterReplay(op_type, soc_version, func);
}
}  // end of namespace optiling
