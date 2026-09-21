/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_LOAD_TS_MEM_MALL_H_
#define GE_GRAPH_LOAD_TS_MEM_MALL_H_

#include <mutex>
#include <unordered_map>
#include <memory>

#include "common/math/math_util.h"
#include "runtime/base.h"
#include "framework/common/debug/ge_log.h"
#include "runtime/mem.h"
#include "graph_metadef/common/ge_common/util.h"

namespace ge {
class TsMemMall {
 public:
  TsMemMall() : TsMemMall(RT_MEMORY_TS) {}
  explicit TsMemMall(const rtMemType_t type) : mem_type_(type) {}

  ~TsMemMall() {
    try {
      for (auto it : mem_store_size_) {
        GE_FREE_RT_LOG(it.second);
      }
      mem_store_size_.clear();
      mem_store_addr_.clear();
    } catch (...) {
      // no processing
    }
  }

  void *Acquire(const int64_t offset, const uint64_t size) {
    constexpr uint32_t kMaxTsMemBlock = 2097152U;   // Max block 2M 2 * 1024 * 1024
    constexpr uint64_t kTsMemAligment = 64U;   // Malloc for 64 bits align
    constexpr uint64_t kTsMemAlignMask = kTsMemAligment - 1U;
    if (size == 0U) {
      GELOGW("[Check][Param] Acquire mem block failed, size:%" PRIu64, size);
      return nullptr;
    }
    if (CheckUint64AddOverflow(size, kTsMemAlignMask) != SUCCESS) {
      GELOGE(RT_FAILED, "[Check][Param] Acquire mem block failed, size:%" PRIu64 " add align_mask:%" PRIu64
        " will overflow:", size, kTsMemAlignMask);
      return nullptr;
    }
    const uint64_t bytes = (size + kTsMemAlignMask) & static_cast<uint64_t>(~kTsMemAlignMask);
    if (bytes > kMaxTsMemBlock) {
      GELOGW("Acquire TS memory may not physical continuity, size: %" PRIu64, bytes);
    }

    const std::lock_guard<std::mutex> lock(mem_mutex_);
    const auto it = mem_store_size_.find(offset);
    if (it != mem_store_size_.end()) {
      GELOGI("Acquire TS memory: %p, offset: %" PRId64 ", size: %" PRIu64 ", align: %" PRIu64,
        it->second, offset, size, bytes);
      return it->second;
    }

    void *addr = nullptr;
    GE_CHK_RT_EXEC(rtMalloc(&addr, bytes, mem_type_, GE_MODULE_NAME_U16), return nullptr);

    GELOGI("Acquire TS memory: %p, offset: %" PRId64 ", size: %" PRIu64 ", align: %" PRIu64,
      addr, offset, size, bytes);
    mem_store_size_[offset] = addr;
    mem_store_addr_[addr] = offset;
    return addr;
  }

 private:
  std::mutex mem_mutex_;
  std::map<int64_t, void *> mem_store_size_;
  std::map<void *, int64_t> mem_store_addr_;
  rtMemType_t mem_type_;
};
}  // namespace ge
#endif  // GE_GRAPH_LOAD_TS_MEM_MALL_H_
