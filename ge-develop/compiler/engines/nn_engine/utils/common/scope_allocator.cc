/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/scope_allocator.h"
#include "common/aicore_util_attr_define.h"
#include "common/fe_log.h"
#include "graph/utils/attr_utils.h"

namespace fe {
namespace {
const std::string kSkpScopeIdAttr = "_skp_fusion_scope";
}

ScopeAllocator::ScopeAllocator() : scope_id_(0), neg_scope_id_(0), fixpipe_scope_id_(0), sk_scope_id_(0) {}

ScopeAllocator::~ScopeAllocator() {}

ScopeAllocator& ScopeAllocator::Instance() {
  static ScopeAllocator scope_allocator;
  return scope_allocator;
}

int64_t ScopeAllocator::AllocateScopeId() {
  ++scope_id_;
  return scope_id_;
}

int64_t ScopeAllocator::GetCurrentScopeId() const {
  return scope_id_;
}

void ScopeAllocator::SetCurrentScopeId(const int64_t &scope_id) {
  scope_id_ = scope_id;
}

int64_t ScopeAllocator::AllocateNegScopeId() {
  --neg_scope_id_;
  return neg_scope_id_;
}

int64_t ScopeAllocator::GetCurrentNegScopeId() const {
  return neg_scope_id_;
}

void ScopeAllocator::SetCurrentNegScopeId(const int64_t &neg_scope_id) {
  neg_scope_id_ = neg_scope_id;
}

int64_t ScopeAllocator::AllocateFixpipeScopeId() {
  ++fixpipe_scope_id_;
  return fixpipe_scope_id_;
}

int64_t ScopeAllocator::GetCurrentFixpipeScopeId() const {
  return fixpipe_scope_id_;
}

void ScopeAllocator::SetFixpipeCurrentScopeId(const int64_t &fixpipe_scope_id) {
  fixpipe_scope_id_ = fixpipe_scope_id;
}

int64_t ScopeAllocator::AllocateSkpScopeId() {
  ++sk_scope_id_;
  return sk_scope_id_;
}

bool ScopeAllocator::HasScopeAttr(ge::ConstOpDescPtr op_desc) {
  if (op_desc == nullptr) {
    return false;
  }

  return op_desc->HasAttr(SCOPE_ID_ATTR);
}

bool ScopeAllocator::GetScopeAttr(ge::ConstOpDescPtr op_desc, int64_t &scope_id) {
  if (op_desc == nullptr) {
    return false;
  }

  return ge::AttrUtils::GetInt(op_desc, SCOPE_ID_ATTR, scope_id);
}

bool ScopeAllocator::SetScopeAttr(const ge::OpDescPtr op_desc, const int64_t &scope_id) {
  if (op_desc == nullptr) {
    REPORT_FE_ERROR("[SubGraphOpt][PostProcess][SetScopeAttr] opdef is nullptr.");
    return false;
  }

  return ge::AttrUtils::SetInt(op_desc, SCOPE_ID_ATTR, scope_id);
}

bool ScopeAllocator::HasL1ScopeAttr(ge::ConstOpDescPtr op_desc) {
  if (op_desc == nullptr) {
    return false;
  }

  return op_desc->HasAttr(L1_SCOPE_ID_ATTR);
}

bool ScopeAllocator::GetL1ScopeAttr(ge::ConstOpDescPtr op_desc, int64_t &scope_id) {
  if (op_desc == nullptr) {
    return false;
  }

  return ge::AttrUtils::GetInt(op_desc, L1_SCOPE_ID_ATTR, scope_id);
}

bool ScopeAllocator::SetL1ScopeAttr(const ge::OpDescPtr op_desc, const int64_t &scope_id) {
  if (op_desc == nullptr) {
    REPORT_FE_ERROR("[SubGraphOpt][PostProcess][SetL1ScopeAttr] opdef is nullptr.");
    return false;
  }

  return ge::AttrUtils::SetInt(op_desc, L1_SCOPE_ID_ATTR, scope_id);
}

bool ScopeAllocator::HasSkpScopeAttr(ge::ConstOpDescPtr op_desc) {
  if (op_desc == nullptr) {
    return false;
  }

  return op_desc->HasAttr(kSkpScopeIdAttr);
}

bool ScopeAllocator::GetSkpScopeAttr(ge::ConstOpDescPtr op_desc, int64_t &sk_scope_id) {
  if (op_desc == nullptr) {
    return false;
  }

  return ge::AttrUtils::GetInt(op_desc, kSkpScopeIdAttr, sk_scope_id);
}

bool ScopeAllocator::SetSkpScopeAttr(const ge::OpDescPtr op_desc, const int64_t &sk_scope_id) {
  if (op_desc == nullptr) {
    FE_LOGI("Setting skp scope id unsuccessful, op description is nullptr.");
    return false;
  }

  return ge::AttrUtils::SetInt(op_desc, kSkpScopeIdAttr, sk_scope_id);
}

bool ScopeAllocator::GetSuperKernelScope(ge::ConstOpDescPtr op_desc, int64_t &sk_scope_id) {
  if (op_desc == nullptr) {
    return false;
  }
  return ge::AttrUtils::GetInt(op_desc, kAscendcSuperKernelScope, sk_scope_id);
}
}  // namespace fe
