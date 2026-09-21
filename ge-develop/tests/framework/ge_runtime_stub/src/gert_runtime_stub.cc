/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "stub/gert_runtime_stub.h"
#include <algorithm>
#include "exe_graph/runtime/tiling_context.h"
#include "faker/nodes_faker_for_exe.h"
#include "faker/base_node_exe_faker.h"
#include "faker/fake_kernel_definitions.h"
#include "common/checker.h"
#include "faker/nodes_faker_for_exe.h"
#include "kernel/memory/caching_mem_allocator.h"
#include "register/kernel_registry.h"

namespace gert {
GertRuntimeStub::GertRuntimeStub(std::unique_ptr<RuntimeStubImpl> rts_runtime_impl, const bool reset_slog_stub,
                                 std::unique_ptr<AclRuntimeStubImpl> acl_runtime_impl)
    : rts_runtime_stub_(std::move(rts_runtime_impl)),
      acl_runtime_stub_(std::move(acl_runtime_impl)),
      need_reset_slog_stub_(reset_slog_stub),
      slog_stub_(std::make_shared<SlogStubImpl>()) {
  ge::RuntimeStub::Install(rts_runtime_stub_.get());
  if (acl_runtime_stub_ != nullptr) {
    ge::AclRuntimeStub::Install(acl_runtime_stub_.get());
  }
  if (need_reset_slog_stub_) {
    ge::SlogStub::SetInstance(slog_stub_);
  }
}

GertRuntimeStub::~GertRuntimeStub() {
  if (need_reset_slog_stub_) {
    // 若GertRuntimeStub在构造时替换了SlogStub实例, 析够时需要恢复
    ge::SlogStub::SetInstance(nullptr);
  }
  Clear();
  ge::RuntimeStub::UnInstall(rts_runtime_stub_.get());
  if (acl_runtime_stub_ != nullptr) {
    ge::AclRuntimeStub::UnInstall(acl_runtime_stub_.get());
  }
}

bool GertRuntimeStub::CheckLaunchWhenStubTiling() {
  for (auto &iter : rts_runtime_stub_->GetLaunchWithHandleArgs()) {
    for (auto launch_arg : iter.second) {
      if (*launch_arg->GetArgsTilingData<uint64_t>() != 100) {
        return false;
      }
    }
  }
  if (acl_runtime_stub_ != nullptr) {
    for (auto &iter : acl_runtime_stub_->GetLaunchWithHandleArgs()) {
      for (auto launch_arg : iter.second) {
        if (*launch_arg->GetArgsTilingData<uint64_t>() != 100) {
          return false;
        }
      }
    }
  }
  return true;
}
GertRuntimeStub &GertRuntimeStub::StubByNodeTypes(const std::vector<std::string> &node_types,
                                                  const std::vector<int> &placements) {
  for (size_t i = 0U; i < node_types.size(); i++) {
    NodesFakerForExe::FakeNode(node_types[i], this, (i < placements.size() ? placements[i] : -1));
  }
  return *this;
}

void GertRuntimeStub::AddCtxFaker(const std::string &type, std::shared_ptr<BaseNodeExeFaker> faker) {
  NodesFakerForExe::PushCtxFaker(type, std::move(faker));
  ctx_mocking_ops_.emplace_back(type);
}

void GertRuntimeStub::ClearCtxFakers() {
  for (auto &type : ctx_mocking_ops_) {
    NodesFakerForExe::PopCtxFaker(type);
  }
  ctx_mocking_ops_.clear();
}
}  // namespace gert
