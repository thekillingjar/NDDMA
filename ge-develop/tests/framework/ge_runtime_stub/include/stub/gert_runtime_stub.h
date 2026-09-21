/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef D0336AC57DE2451BBA4F946AEB8285B5_H
#define D0336AC57DE2451BBA4F946AEB8285B5_H

#include "kernel_stub.h"
#include "converter_stub.h"
#include "ge_fake_launch_args.h"
#include "runtime_stub_impl.h"
#include "acl_runtime_stub_impl.h"
#include "slog_stub_impl.h"
#include "task_info_registry_stub.h"
#include <list>
#include <map>
#include <memory>

namespace gert {
class BaseNodeExeFaker;
// RuntimeStub 已经有这个名字了，代表的是rts的runtime stub
class GertRuntimeStub {
 public:
  GertRuntimeStub(bool reset_slog_stub = true)
      : GertRuntimeStub(std::make_unique<RuntimeStubImpl>(), reset_slog_stub, std::make_unique<AclRuntimeStubImpl>()) {
    Clear();
  }
  explicit GertRuntimeStub(std::unique_ptr<RuntimeStubImpl> rts_runtime_impl, bool reset_slog_stub = true, std::unique_ptr<AclRuntimeStubImpl> acl_runtime_impl = nullptr);
  ~GertRuntimeStub();
  bool CheckLaunchWhenStubTiling();
  void Clear() {
    ClearCtxFakers();
    rts_runtime_stub_->Clear();
    if (acl_runtime_stub_ != nullptr) {
      acl_runtime_stub_->Clear();
    }
    converter_stub_.Clear();
    kernel_stub_.Clear();
  }

  KernelStub &GetKernelStub() {
    return kernel_stub_;
  }

  RuntimeStubImpl &GetRtsRuntimeStub() {
    return *rts_runtime_stub_;
  }

  AclRuntimeStubImpl &GetAclRuntimeStub() {
    return *acl_runtime_stub_;
  }

  const RuntimeStubImpl &GetRtsRuntimeStub() const {
    return *rts_runtime_stub_;
  }

  const AclRuntimeStubImpl &GetAclRuntimeStub() const {
    return *acl_runtime_stub_;
  }

  ConverterStub &GetConverterStub() {
    return converter_stub_;
  }
  const ConverterStub &GetConverterStub() const {
    return converter_stub_;
  }

  SlogStubImpl &GetSlogStub() {
    return *slog_stub_;
  }
  const SlogStubImpl &GetSlogStub() const {
    return *slog_stub_;
  }

  ge::TaskInfoRegistryStub &GetTaskInfoFactoryStub() {
    return task_info_factory_stub_;
  }
  const ge::TaskInfoRegistryStub &GetTaskInfoFactoryStub() const {
    return task_info_factory_stub_;
  }

  GertRuntimeStub &StubByNodeTypes(const std::vector<std::string> &node_types, const std::vector<int> &placements = {});

  void AddCtxFaker(const std::string &type, std::shared_ptr<BaseNodeExeFaker> faker);
  void ClearCtxFakers();

 private:
  KernelStub kernel_stub_;
  ConverterStub converter_stub_;
  std::unique_ptr<RuntimeStubImpl> rts_runtime_stub_;
  std::unique_ptr<AclRuntimeStubImpl> acl_runtime_stub_;
  std::vector<std::string> ctx_mocking_ops_;
  bool need_reset_slog_stub_;
  std::shared_ptr<SlogStubImpl> slog_stub_;
  ge::TaskInfoRegistryStub task_info_factory_stub_;
};
}  // namespace gert
#endif
