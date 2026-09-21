/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "register/op_def.h"
#include "op_def_impl.h"
#include "framework/common/debug/ge_log.h"

namespace ops {
OpHostCPUDef::OpHostCPUDef() : impl_(new(std::nothrow) OpHostCPUDefImpl) {
    this->Engine("DNN_VM_HOST_CPU")
    .FlagPartial(false)
    .ComputeCost("100")
    .FlagAsync(false)
    .OpKernelLib("HOSTCPUKernel")
    .KernelSo("libcpu_kernels.so")
    .FunctionName("RunCpuKernel")
    .UserDefined(false);
}

OpHostCPUDef::OpHostCPUDef(const OpHostCPUDef &hostcpu_def) : impl_(new(std::nothrow) OpHostCPUDefImpl) {
  this->impl_->cfg_keys = hostcpu_def.impl_->cfg_keys;
  this->impl_->cfg_info = hostcpu_def.impl_->cfg_info;
}

OpHostCPUDef::~OpHostCPUDef() = default;

OpHostCPUDef &OpHostCPUDef::operator=(const OpHostCPUDef &hostcpu_def) {
  if (this != &hostcpu_def) {
    *this->impl_ = *hostcpu_def.impl_;
  }
  return *this;
}

OpHostCPUDef &OpHostCPUDef::ExtendCfgInfo(const char *key, const char *value) {
    this->AddCfgItem(key, value);
    return *this;
}

void OpHostCPUDef::AddCfgItem(const char *key, const char *value) {
  auto it = this->impl_->cfg_info.find(key);
  if (it == this->impl_->cfg_info.cend()) {
    this->impl_->cfg_keys.emplace_back(key);
  } else {
    this->impl_->cfg_info.erase(key);
  }
  this->impl_->cfg_info.emplace(key, value);
}

std::vector<ge::AscendString> &OpHostCPUDef::GetCfgKeys(void) {
  return this->impl_->cfg_keys;
}

std::map<ge::AscendString, ge::AscendString> &OpHostCPUDef::GetCfgInfo(void) {
  return this->impl_->cfg_info;
}

ge::AscendString &OpHostCPUDef::GetConfigValue(const char *key) {
  return this->impl_->cfg_info[key];
}

OpHostCPUDef &OpHostCPUDef::Engine(const char *value) {
    this->AddCfgItem("opInfo.engine", value);
    return *this;
}

OpHostCPUDef &OpHostCPUDef::FlagPartial(bool flag) {
    this->AddCfgItem("opInfo.flagPartial", flag ? "True" : "False");
    return *this;
}

OpHostCPUDef &OpHostCPUDef::ComputeCost(const char *value) {
    this->AddCfgItem("opInfo.computeCost", value);
    return *this;
}

OpHostCPUDef &OpHostCPUDef::FlagAsync(bool flag) {
    this->AddCfgItem("opInfo.flagAsync", flag ? "True" : "False");
    return *this;
}

OpHostCPUDef &OpHostCPUDef::OpKernelLib(const char *value) {
    this->AddCfgItem("opInfo.opKernelLib", value);
    return *this;
}

OpHostCPUDef &OpHostCPUDef::KernelSo(const char *value) {
    this->AddCfgItem("opInfo.kernelSo", value);
    return *this;
}

OpHostCPUDef &OpHostCPUDef::FunctionName(const char *value) {
    this->AddCfgItem("opInfo.functionName", value);
    return *this;
}

OpHostCPUDef &OpHostCPUDef::UserDefined(bool flag) {
    this->AddCfgItem("opInfo.userDefined", flag ? "True" : "False");
    return *this;
}
}