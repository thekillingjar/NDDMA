/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_KERNEL_MEMORY_UTIL_VERSION_BLOCKS_H_
#define AIR_CXX_RUNTIME_V2_KERNEL_MEMORY_UTIL_VERSION_BLOCKS_H_
#include <functional>
#include "multi_stream_mem_block.h"
#include "common/checker.h"

namespace gert {
namespace memory {
struct VersionBlock {
  int64_t version;
  MultiStreamMemBlock *block;
};
struct StreamedVersionBlock {
  VersionBlock version_block;
  // bit位，已发送标记，未发送时为0，已发送为1
  uint64_t sent_flags;
};

class BaseVersionBlocks {
 public:
  using FindNextPolicy = void (BaseVersionBlocks::*)(std::list<StreamedVersionBlock>::iterator &);

  static uint64_t ToBit(int64_t stream_id) {
    return static_cast<uint64_t>(1) << static_cast<uint64_t>(stream_id);
  }
  static uint64_t ToAllBit(int64_t stream_num, int64_t stream_id) {
    return (ToBit(stream_num) - 1U) & ~ToBit(stream_id);
  }

  BaseVersionBlocks(int64_t stream_id, uint64_t dst_stream_bit, uint64_t all_sent_bits,
                    std::list<StreamedVersionBlock> &blocks)
      : stream_id_(stream_id), dst_stream_bit_(dst_stream_bit), all_sent_bits_(all_sent_bits), blocks_(blocks) {}

  void FindNext(std::list<StreamedVersionBlock>::iterator &iter) {
    while (iter != blocks_.end()) {
      if (iter->version_block.version != iter->version_block.block->GetVersion()) {
        GELOGD("Drop local-recycle block %p on stream %" PRId64 ", the block has a newer version", stream_id_,
               iter->version_block.block);
        iter = blocks_.erase(iter);
        continue;
      }
      if (iter->sent_flags == all_sent_bits_) {
        iter = blocks_.erase(iter);
        continue;
      }
      if (iter->sent_flags & dst_stream_bit_) {
        ++iter;
        continue;
      }
      iter->sent_flags |= dst_stream_bit_;
      return;
    }
  }

  void FindNextForAll(std::list<StreamedVersionBlock>::iterator &iter) {
    while (iter != blocks_.end()) {
      if (iter->version_block.version != iter->version_block.block->GetVersion()) {
        GELOGD("Drop local-recycle block %p on stream %" PRId64 ", the block has a newer version", stream_id_,
               iter->version_block.block);
        iter = blocks_.erase(iter);
        continue;
      }
      if (iter->sent_flags == all_sent_bits_) {
        iter = blocks_.erase(iter);
        continue;
      }
      iter->sent_flags |= dst_stream_bit_;
      return;
    }
  }

 private:
  int64_t stream_id_;
  uint64_t dst_stream_bit_;
  uint64_t all_sent_bits_;

 protected:
  std::list<StreamedVersionBlock> &blocks_;
};

template <BaseVersionBlocks::FindNextPolicy F>
class VersionBlocks : public BaseVersionBlocks {
 public:
  VersionBlocks(int64_t stream_id, uint64_t dst_stream_bit, int64_t stream_num, std::list<StreamedVersionBlock> &blocks)
      : BaseVersionBlocks(stream_id, dst_stream_bit, ToAllBit(stream_num, stream_id), blocks) {}

  std::list<StreamedVersionBlock>::iterator Begin() {
    auto iter = blocks_.begin();
    (this->*F)(iter);
    return iter;
  }
  std::list<StreamedVersionBlock>::iterator End() {
    return blocks_.end();
  }
  void Next(std::list<StreamedVersionBlock>::iterator &iter) {
    ++iter;
    (this->*F)(iter);
  }
};
}  // namespace memory
}  // namespace gert
#endif  // AIR_CXX_RUNTIME_V2_KERNEL_MEMORY_UTIL_VERSION_BLOCKS_H_
