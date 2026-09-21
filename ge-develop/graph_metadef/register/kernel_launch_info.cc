/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/kernel_launch_info.h"
#include "graph_metadef/graph/debug/ge_util.h"
#include "kernel_launch_info_impl.h"
#include "common/checker.h"

namespace ge {
KernelLaunchInfo::~KernelLaunchInfo() {}
KernelLaunchInfo::KernelLaunchInfo(const KernelLaunchInfo &other) {
  impl_ = ComGraphMakeUnique<KernelLaunchInfoImpl>();
  if ((other.impl_ != nullptr) && (impl_ != nullptr)) {
    *impl_ = *other.impl_;
  }
}
KernelLaunchInfo::KernelLaunchInfo(KernelLaunchInfo &&other) noexcept {
  impl_ = std::move(other.impl_);
}
KernelLaunchInfo &KernelLaunchInfo::operator=(const KernelLaunchInfo &other) {
  if (&other != this) {
    impl_ = ComGraphMakeUnique<KernelLaunchInfoImpl>();
    if ((other.impl_ != nullptr) && (impl_ != nullptr)) {
      *impl_ = *other.impl_;
    }
  }
  return *this;
}
KernelLaunchInfo &KernelLaunchInfo::operator=(KernelLaunchInfo &&other) noexcept {
  if (&other != this) {
    impl_ = std::move(other.impl_);
  }
  return *this;
}

KernelLaunchInfo::KernelLaunchInfo(KernelLaunchInfoImplPtr &&impl) : impl_(std::move(impl)) {}

KernelLaunchInfo KernelLaunchInfo::LoadFromData(const gert::ExeResGenerationContext *context,
    const std::vector<uint8_t> &data) {
  return KernelLaunchInfo(KernelLaunchInfoImpl::LoadFromData(context, data));
}
KernelLaunchInfo KernelLaunchInfo::CreateAicpuKfcTask(const gert::ExeResGenerationContext *context,
    const char *so_name, const char *kernel_name) {
  return KernelLaunchInfo(KernelLaunchInfoImpl::CreateAicpuKfcTask(context, so_name, kernel_name));
}

KernelLaunchInfo KernelLaunchInfo::CreateHcomRecordTask(const gert::ExeResGenerationContext *context,
    const char *group_name) {
  return KernelLaunchInfo(KernelLaunchInfoImpl::CreateHcomRecordTask(context, group_name));
}
KernelLaunchInfo KernelLaunchInfo::CreateHcomWaitTask(const gert::ExeResGenerationContext *context,
    const char *group_name) {
  return KernelLaunchInfo(KernelLaunchInfoImpl::CreateHcomWaitTask(context, group_name));
}
KernelLaunchInfo KernelLaunchInfo::CreateFusionTask(
    const gert::ExeResGenerationContext *context,
    const std::vector<KernelLaunchInfo>& sub_tasks) {
    std::vector<const KernelLaunchInfoImpl*> sub_tasks_ref;
    sub_tasks_ref.reserve(sub_tasks.size());

    for (const auto &task : sub_tasks) {
        if (task.impl_) {
            sub_tasks_ref.emplace_back(task.impl_.get());
        }
    }

    return KernelLaunchInfo(
        KernelLaunchInfoImpl::CreateFusionTask(context, sub_tasks_ref));
}
KernelLaunchInfo KernelLaunchInfo::CreateCcuTask(const gert::ExeResGenerationContext *context,
    const std::vector<std::string>& groups) {
  return KernelLaunchInfo(KernelLaunchInfoImpl::CreateCcuTask(context, groups));
}
std::vector<uint8_t> KernelLaunchInfo::Serialize() {
  if (impl_ != nullptr) {
    return impl_->Serialize();
  }
  return {};
}
uint32_t KernelLaunchInfo::GetStreamId() const {
  if (impl_ != nullptr) {
    return impl_->GetStreamId();
  }
  return std::numeric_limits<uint32_t>::max();
}

void KernelLaunchInfo::SetStreamId(uint32_t stream_id) {
  if (impl_ != nullptr) {
    impl_->SetStreamId(stream_id);
  }
}

uint32_t KernelLaunchInfo::GetBlockDim() const {
  if (impl_ != nullptr) {
    return impl_->GetBlockDim();
  }
  return std::numeric_limits<uint32_t>::max();
}

graphStatus KernelLaunchInfo::SetBlockDim(uint32_t block_dim) {
  GE_ASSERT_NOTNULL(impl_);
  return impl_->SetBlockDim(block_dim);
}

const char *KernelLaunchInfo::GetArgsFormat() const {
  if (impl_ != nullptr) {
    return impl_->GetArgsFormat();
  }
  return nullptr;
}
graphStatus KernelLaunchInfo::SetArgsFormat(const char *args_format) {
  GE_ASSERT_NOTNULL(impl_);
  return impl_->SetArgsFormat(args_format);
}

const char *KernelLaunchInfo::GetSoName() const {
  if (impl_ != nullptr) {
    return impl_->GetSoName();
  }
  return nullptr;
}
const char *KernelLaunchInfo::GetKernelName() const {
  if (impl_ != nullptr) {
    return impl_->GetKernelName();
  }
  return nullptr;
}
}