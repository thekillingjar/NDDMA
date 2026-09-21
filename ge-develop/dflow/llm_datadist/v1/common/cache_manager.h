/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_RUNTIME_LLM_ENGINE_COMMON_CACHE_MANAGER_H_
#define AIR_RUNTIME_LLM_ENGINE_COMMON_CACHE_MANAGER_H_

#include <map>
#include <mutex>
#include "llm_datadist/llm_error_codes.h"
#include "acl/acl.h"
// for using rtMemcpyEx with mbuf
#include "runtime/rt_external.h"
#include "common.h"

namespace llm {
using DataCacheKey = std::pair<uint64_t, uint64_t>;  // req id/prefix id, model id
class CacheManager {
 public:
  CacheManager() = default;
  ~CacheManager() = default;
  ge::Status CopyCacheForContinuous(const CacheEntry &src_cache_entry,
                                    const CacheEntry &dst_cache_entry,
                                    const CopyCacheParam &copy_cache_param,
                                    size_t per_device_addr_num,
                                    size_t device_index = 0U);
  ge::Status CopyCacheForBlocks(const CacheEntry &src_cache_entry,
                                const CacheEntry &dst_cache_entry,
                                const CopyCacheParam &copy_cache_param,
                                size_t per_device_addr_num,
                                size_t device_index = 0U);
  ge::Status InitCopyStreams(size_t device_num);
  void DestroyCopyStream(size_t device_index);

private:
  static ge::Status CheckCopyParams(const CacheEntry &src_cache_entry,
                                    const CacheEntry &dst_cache_entry,
                                    const CopyCacheParam &copy_cache_param,
                                    uint64_t &copy_size);
  static rtMemcpyKind_t ResolveCopyKind(CachePlacement src_placement, CachePlacement dst_placement);
  ge::Status EnsureCopyStream(size_t device_index);

  std::mutex copy_mu_;
  std::vector<aclrtStream> copy_streams_;
};
}  // namespace llm

#endif  // AIR_RUNTIME_LLM_ENGINE_COMMON_CACHE_MANAGER_H_
