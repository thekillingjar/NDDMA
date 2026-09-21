/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "exe_graph/lowering/buffer_pool.h"

#include <securec.h>
#include "framework/common/debug/ge_log.h"
#include "graph/utils/math_util.h"
#include "graph/def_types.h"
#include "graph_metadef/graph/debug/ge_util.h"
#include "common/checker.h"

#include "exe_graph/runtime/continuous_buffer.h"

namespace gert {
namespace bg {
namespace {
constexpr size_t kLargeBufSizeThreshold = 1024U * 1024U; // 1M
}
BufferPool::BufId BufferPool::AddBuf(const uint8_t *data, const size_t len) {
  if (len >= kLargeBufSizeThreshold) {
    return AddLargeBuf(std::string(ge::PtrToPtr<uint8_t, char>(data), len));
  }
  return AddBuf(std::string(ge::PtrToPtr<uint8_t, char>(data), len));
}
BufferPool::BufId BufferPool::AddStr(const char *data) {
  size_t len = strlen(data) + 1;
  if (len >= kLargeBufSizeThreshold) {
    return AddLargeBuf(std::string(data, len));
  }
  return AddBuf(std::string(data, len));
}
BufferPool::BufId BufferPool::AddBuf(std::string &&str) {
  auto res = bufs_to_id_.emplace(std::move(str), id_generator_);
  if (res.second) {
    ++id_generator_;
  }
  return res.first->second;
}
BufferPool::BufId BufferPool::AddLargeBuf(std::string &&str) {
  auto id = id_generator_++;
  large_bufs_to_id_.emplace_back(std::move(str), id);
  return id;
}
std::unique_ptr<uint8_t[]> BufferPool::Serialize() const {
  size_t total_size;
  return Serialize(total_size);
}
std::unique_ptr<uint8_t[]> BufferPool::Serialize(size_t &total_size) const {
  total_size = sizeof(ContinuousBuffer);
  const size_t buf_count = id_generator_;
  size_t offset_size;
  size_t text_offset;
  // 申请了n个，但是使用时会用n+1个，多的一个由ContinuousText自带
  if (ge::MulOverflow(sizeof(size_t), buf_count, offset_size)) {
    GE_LOGE("Failed to serialize buffer pool, size overflow, buf num %zu", buf_count);
    return nullptr;
  }
  if (ge::AddOverflow(total_size, offset_size, total_size)) {
    GE_LOGE("Failed to serialize buffer pool, size overflow, buf size %zu", offset_size);
    return nullptr;
  }
  text_offset = total_size;

  std::vector<const std::string *> ids_to_buf(buf_count);
  for (const auto &iter : bufs_to_id_) {
    if (iter.second >= buf_count) {
      return nullptr;
    }
    ids_to_buf[iter.second] = &iter.first;

    if (ge::AddOverflow(total_size, iter.first.size(), total_size)) {
      GE_LOGE("Failed to serialize buffer pool, size overflow, buf size %zu, id %zu", iter.first.size(), iter.second);
      return nullptr;
    }
  }
  for (const auto &iter : large_bufs_to_id_) {
    if (iter.second >= buf_count) {
      return nullptr;
    }
    ids_to_buf[iter.second] = &iter.first;

    if (ge::AddOverflow(total_size, iter.first.size(), total_size)) {
      GE_LOGE("Failed to serialize buffer pool, size overflow, buf size %zu, id %zu", iter.first.size(), iter.second);
      return nullptr;
    }
  }

  auto text_holder = ge::ComGraphMakeUnique<uint8_t[]>(total_size);
  GE_ASSERT_NOTNULL(text_holder);

  auto text = ge::PtrToPtr<uint8_t, ContinuousBuffer>(text_holder.get());
  text->num_ = buf_count;
  text->reserved_ = 0;
  size_t i = 0;
  for (; i < buf_count; ++i) {
    const auto buf = ids_to_buf[i];
    if (buf == nullptr) {
      GELOGE(ge::FAILED, "Failed to serialize text pool, miss buf id %zu", i);
      return nullptr;
    }
    const auto ret = ge::GeMemcpy(text_holder.get() + text_offset, total_size - text_offset,
        ge::PtrToPtr<char, uint8_t>(buf->data()), buf->size());
    GE_ASSERT_TRUE((ret == ge::SUCCESS), "memcpy_s failed, copy size is %zu, dst size is %zu",
        buf->size(), total_size - text_offset);
    text->offsets_[i] = text_offset;
    text_offset += buf->size();
  }
  text->offsets_[i] = text_offset;

  return text_holder;
}
const char *BufferPool::GetBufById(const BufId id) const {
  for (const auto &buf_and_id : bufs_to_id_) {
    if (buf_and_id.second == id) {
      return buf_and_id.first.c_str();
    }
  }
  for (const auto &buf_and_id : large_bufs_to_id_) {
    if (buf_and_id.second == id) {
      return buf_and_id.first.c_str();
    }
  }
  return nullptr;
}
size_t BufferPool::GetSize() const {
  return id_generator_;
}
}  // namespace bg
}  // namespace gert
