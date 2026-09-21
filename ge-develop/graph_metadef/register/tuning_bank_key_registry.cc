/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/tuning_bank_key_registry.h"
#include "framework/common/debug/ge_log.h"

namespace tuningtiling {
OpBankKeyFuncInfo::OpBankKeyFuncInfo(const ge::AscendString &optype) : optype_(optype) {}

OpBankKeyFuncInfoV2::OpBankKeyFuncInfoV2(const ge::AscendString &optypeV2) : optypeV2_(optypeV2) {}

// v1兼容老版本om
void OpBankKeyFuncInfo::SetOpConvertFunc(const OpBankKeyConvertFun &convert_func) {
  convert_func_ = convert_func;
}

void OpBankKeyFuncInfo::SetOpParseFunc(const OpBankParseFun &parse_func) {
  parse_func_ = parse_func;
}

void OpBankKeyFuncInfo::SetOpLoadFunc(const OpBankLoadFun &load_func) {
  load_func_ = load_func;
}


// v2
void OpBankKeyFuncInfoV2::SetOpConvertFuncV2(const OpBankKeyConvertFunV2 &convert_funcV2) {
  convert_funcV2_ = convert_funcV2;
}

void OpBankKeyFuncInfoV2::SetOpParseFuncV2(const OpBankParseFunV2 &parse_funcV2) {
  parse_funcV2_ = parse_funcV2;
}

void OpBankKeyFuncInfoV2::SetOpLoadFuncV2(const OpBankLoadFunV2 &load_funcV2) {
  load_funcV2_ = load_funcV2;
}

// v1兼容老版本om
const OpBankKeyConvertFun& OpBankKeyFuncInfo::OpBankKeyFuncInfo::GetBankKeyConvertFunc() const {
  return convert_func_;
}

const OpBankParseFun& OpBankKeyFuncInfo::GetBankKeyParseFunc() const {
  return parse_func_;
}

const OpBankLoadFun& OpBankKeyFuncInfo::GetBankKeyLoadFunc() const {
  return load_func_;
}

// v2
const OpBankKeyConvertFunV2& OpBankKeyFuncInfoV2::OpBankKeyFuncInfoV2::GetBankKeyConvertFuncV2() const {
  return convert_funcV2_;
}

const OpBankParseFunV2& OpBankKeyFuncInfoV2::GetBankKeyParseFuncV2() const {
  return parse_funcV2_;
}

const OpBankLoadFunV2& OpBankKeyFuncInfoV2::GetBankKeyLoadFuncV2() const {
  return load_funcV2_;
}

// v1兼容老版本om
std::unordered_map<ge::AscendString, OpBankKeyFuncInfo> &OpBankKeyFuncRegistry::RegisteredOpFuncInfo() {
  static std::unordered_map<ge::AscendString, OpBankKeyFuncInfo> op_func_map;
  return op_func_map;
}

// v2
std::unordered_map<ge::AscendString, OpBankKeyFuncInfoV2> &OpBankKeyFuncRegistryV2::RegisteredOpFuncInfoV2() {
  static auto* op_func_mapV2 = new (std::nothrow) std::unordered_map<ge::AscendString, OpBankKeyFuncInfoV2>();
  return *op_func_mapV2;
}

extern "C" void _ZN12tuningtiling21OpBankKeyFuncRegistryC1ERKN2ge12AscendStringERKSt8functionIFbRKSt10shared_ptrIvEmRN15ascend_nlohmann10basic_jsonISt3mapSt6vectorSsblmdSaNSA_14adl_serializerESD_IhSaIhEEEEEERKS5_IFbRS7_RmRKSH_EE() {}

extern "C" void _ZN12tuningtiling21OpBankKeyFuncRegistryC1ERKN2ge12AscendStringERKSt8functionIFbRKSt10shared_ptrIvEmRN15ascend_nlohmann16json_abi_v3_11_210basic_jsonISt3mapSt6vectorSsblmdSaNSB_14adl_serializerESE_IhSaIhEEEEEERKS5_IFbRS7_RmRKSI_EE() {}
// v1兼容老版本om
OpBankKeyFuncRegistry::OpBankKeyFuncRegistry(const ge::AscendString &optype, const OpBankKeyConvertFun &convert_func) {
  auto &op_func_map = RegisteredOpFuncInfo();
  const auto iter = op_func_map.find(optype);
  if (iter == op_func_map.cend()) {
    OpBankKeyFuncInfo op_func_info(optype);
    op_func_info.SetOpConvertFunc(convert_func);
    (void)op_func_map.emplace(optype, op_func_info);
  } else {
    iter->second.SetOpConvertFunc(convert_func);
  }
}

OpBankKeyFuncRegistry::OpBankKeyFuncRegistry(const ge::AscendString &optype,
                                             const OpBankParseFun &parse_func, const OpBankLoadFun &load_func) {
  auto &op_func_map = RegisteredOpFuncInfo();
  const auto iter = op_func_map.find(optype);
  if (iter == op_func_map.cend()) {
    OpBankKeyFuncInfo op_func_info(optype);
    op_func_info.SetOpParseFunc(parse_func);
    op_func_info.SetOpLoadFunc(load_func);
    (void)op_func_map.emplace(optype, op_func_info);
  } else {
    iter->second.SetOpParseFunc(parse_func);
    iter->second.SetOpLoadFunc(load_func);
  }
}

// v2接口
OpBankKeyFuncRegistryV2::OpBankKeyFuncRegistryV2(const ge::AscendString &optype, const OpBankKeyConvertFunV2 &convert_funcV2) {
  auto &op_func_mapV2 = RegisteredOpFuncInfoV2();
  const auto iter = op_func_mapV2.find(optype);
  if (iter == op_func_mapV2.cend()) {
    OpBankKeyFuncInfoV2 op_func_info(optype);
    op_func_info.SetOpConvertFuncV2(convert_funcV2);
    (void)op_func_mapV2.emplace(optype, op_func_info);
  } else {
    iter->second.SetOpConvertFuncV2(convert_funcV2);
  }
}

OpBankKeyFuncRegistryV2::OpBankKeyFuncRegistryV2(const ge::AscendString &optype, const OpBankParseFunV2 &parse_funcV2,
                                                 const OpBankLoadFunV2 &load_funcV2) {
  auto &op_func_map = RegisteredOpFuncInfoV2();
  const auto iter = op_func_map.find(optype);
  if (iter == op_func_map.cend()) {
    OpBankKeyFuncInfoV2 op_func_info(optype);
    op_func_info.SetOpParseFuncV2(parse_funcV2);
    op_func_info.SetOpLoadFuncV2(load_funcV2);
    (void)op_func_map.emplace(optype, op_func_info);
  } else {
    iter->second.SetOpParseFuncV2(parse_funcV2);
    iter->second.SetOpLoadFuncV2(load_funcV2);
  }
}
}  // namespace tuningtiling
