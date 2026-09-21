/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSION_ENGINE_INC_COMMON_SCOPE_ALLOCATOR_H_
#define FUSION_ENGINE_INC_COMMON_SCOPE_ALLOCATOR_H_

#include <atomic>
#include "graph/op_desc.h"

namespace fe {
class ScopeAllocator {
 public:
  static ScopeAllocator& Instance();

  ScopeAllocator(const ScopeAllocator& in) = delete;
  ScopeAllocator& operator = (const ScopeAllocator& in) = delete;

  int64_t AllocateScopeId();
  int64_t GetCurrentScopeId() const;
  void SetCurrentScopeId(const int64_t &scope_id);
  int64_t AllocateNegScopeId();
  int64_t GetCurrentNegScopeId() const;
  void SetCurrentNegScopeId(const int64_t &neg_scope_id);
  int64_t AllocateFixpipeScopeId();
  int64_t GetCurrentFixpipeScopeId() const;
  void SetFixpipeCurrentScopeId(const int64_t &fixpipe_scope_id);
  int64_t AllocateSkpScopeId();
  
  static bool HasScopeAttr(ge::ConstOpDescPtr op_desc);
  static bool GetScopeAttr(ge::ConstOpDescPtr op_desc, int64_t &scope_id);
  static bool SetScopeAttr(const ge::OpDescPtr op_desc, const int64_t &scope_id);
  static bool HasL1ScopeAttr(ge::ConstOpDescPtr op_desc);
  static bool GetL1ScopeAttr(ge::ConstOpDescPtr op_desc, int64_t &scope_id);
  static bool SetL1ScopeAttr(const ge::OpDescPtr op_desc, const int64_t &scope_id);

  static bool HasSkpScopeAttr(ge::ConstOpDescPtr op_desc);
  static bool GetSkpScopeAttr(ge::ConstOpDescPtr op_desc, int64_t &sk_scope_id);
  static bool SetSkpScopeAttr(const ge::OpDescPtr op_desc, const int64_t &sk_scope_id);
  static bool GetSuperKernelScope(ge::ConstOpDescPtr op_desc, int64_t &sk_scope_id);

 private:
  ScopeAllocator();
  virtual ~ScopeAllocator();

  std::atomic<int64_t> scope_id_;
  std::atomic<int64_t> neg_scope_id_;
  std::atomic<int64_t> fixpipe_scope_id_;
  std::atomic<int64_t> sk_scope_id_;
};
}  // namespace fe
#endif  // FUSION_ENGINE_INC_COMMON_SCOPE_ALLOCATOR_H_
