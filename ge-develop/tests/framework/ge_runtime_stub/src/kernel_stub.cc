/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "stub/kernel_stub.h"

#include <cstdint>
#include "stub/ge_fake_launch_args.h"
#include "exe_graph/runtime/tiling_context.h"
#include "stub/converter_stub.h"
#include "kernel/common_kernel_impl/tiling.h"

namespace gert {
namespace {
UINT32 FakeTiling(gert::KernelContext *context) {
  const auto input_num = context->GetInputNum();
  GE_ASSERT(input_num > 2);
  auto fwk_data = context->GetInputPointer<kernel::TilingFwkData>(input_num - 3);
  GE_ASSERT_NOTNULL(fwk_data);
  auto launch_arg = fwk_data->launch_arg;
  GE_ASSERT_NOTNULL(launch_arg);
  auto tiling_data_av = context->GetOutput(TilingContext::kOutputTilingData);
  GE_ASSERT_NOTNULL(tiling_data_av);
  auto launch_arg_av = context->GetOutput(static_cast<size_t>(kernel::TilingExOutputIndex::kRtArg));
  GE_ASSERT_NOTNULL(launch_arg_av);
  tiling_data_av->Set(&launch_arg->GetTilingData(), nullptr);
  launch_arg_av->Set(launch_arg, nullptr);
  auto tiling_context = reinterpret_cast<gert::TilingContext *>(context);
  auto tiling_data = tiling_context->GetTilingData<uint64_t>();
  GE_ASSERT_NOTNULL(tiling_data);
  tiling_context->SetTilingKey(0);
  tiling_context->SetBlockDim(32);
  auto workspaces = tiling_context->GetWorkspaceSizes(1);
  workspaces[0] = 4096;
  *tiling_data = 100;
  return 0;
}

ge::graphStatus FakeTilingParse(gert::KernelContext *context) {
  return 0;
}

UINT32 FakeGenerateSqeAndLaunchTask(gert::KernelContext *context) {
  return 0;
}

UINT32 FakeCalcDvppWorkSpaceSize(gert::KernelContext *context) {
  return 0;
}
UINT32 FakeAllocDvppWorkSpaceMem(gert::KernelContext *context) {
  return 0;
}
}  // namespace

bool KernelStub::SetUp(const std::string &kernel_name, KernelRegistry::KernelFunc stub_run_func) {
  KernelRegistry::KernelFuncs funcs;

  auto kernel_funcs = KernelRegistry::GetInstance().FindKernelFuncs(kernel_name);
  if (kernel_funcs != nullptr) {
    funcs = *kernel_funcs;
    funcs.run_func = stub_run_func;
  } else {
    funcs = {stub_run_func, nullptr, nullptr};
  }
  SetUp(kernel_name, funcs);
  return true;
}
KernelStub &KernelStub::SetUp(const std::string &kernel_name, const KernelRegistry::KernelFuncs &funcs) {
  StubWhenNeed();
  KernelRegistry::GetInstance().RegisterKernel(kernel_name, {funcs, ""});
  return *this;
}
KernelStub &KernelStub::StubTiling() {
  SetUp("TilingParse", FakeTilingParse);
  SetUp("Tiling", FakeTiling);
  SetUp("CacheableTiling", FakeTiling);
  return *this;
}

KernelStub &KernelStub::StubTilingWithCustomTiling(KernelRegistry::KernelFunc stub_run_func) {
  SetUp("TilingParse", FakeTilingParse);
  SetUp("Tiling", stub_run_func);
  SetUp("CacheableTiling", stub_run_func);
  return *this;
}

KernelStub &KernelStub::StubGenerateSqeAndLaunchTask() {
  SetUp("GenerateSqeAndLaunchTask", FakeGenerateSqeAndLaunchTask);
  return *this;
}

KernelStub &KernelStub::StubCalcDvppWorkSpaceSize() {
  SetUp("CalcDvppWorkSpaceSize", FakeCalcDvppWorkSpaceSize);
  return *this;
}

KernelStub &KernelStub::StubAllocDvppWorkSpaceMem() {
  SetUp("AllocBatchHbm", FakeAllocDvppWorkSpaceMem);
  return *this;
}

KernelStub::~KernelStub() {
  Clear();
}
void KernelStub::Clear() {
  KernelRegistry::ReplaceKernelRegistry(nullptr);
  stub_registry_ = nullptr;
}
void KernelStub::StubWhenNeed() {
  if (stub_registry_ == nullptr) {
    stub_registry_ = std::make_shared<StubKernelRegistry>(KernelRegistryImpl::GetInstance());
    KernelRegistry::ReplaceKernelRegistry(stub_registry_);
  }
}
void KernelStub::AllKernelRegisteredAndSuccess(const std::set<std::string>& exclude_types) {
  StubWhenNeed();
  stub_registry_->always_return_success_kernel = true;
  stub_registry_->always_return_success_kernel_excluded = exclude_types;
}
}  // namespace gert
