/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef METADEF_CXX_MIF_H_
#define METADEF_CXX_MIF_H_
#include <cstdint>
#include <limits>
#include "graph/small_vector.h"
namespace gert {
namespace memory {
constexpr size_t kDefaultMaxStreamNum = 8U;
/**
 * Multi-stream Independent-maintained Flags
 * 本类仅供内部使用，为了性能考虑，避免冗余校验，public方法不做stream合法性校验，调用者需要保证stream id合法，不会出现越界等错误
 */
class Mif {
 public:
  void ReInit(size_t stream_num) {
    stream_ids_to_bits_.clear();
    stream_ids_to_bits_.resize(stream_num, 0U);
  }

  /**
   * 在观察流`maintained_stream`上，`stream`置位了这块资源
   * @param maintained_stream
   * @param stream
   * @return
   */
  void Set(int64_t maintained_stream, int64_t stream) {
    stream_ids_to_bits_[maintained_stream] |= ToBit(stream);
  }
  /**
   * 在所有观察流上，使`stream`置位这块资源
   * @param stream
   */
  void SetAll(int64_t stream) {
    const auto flag = ToBit(stream);
    for (auto &occupy_flags : stream_ids_to_bits_) {
      occupy_flags |= flag;
    }
  }
  /**
   * 在观察流`maintained_stream`上，将所有流置位
   * @param maintained_stream
   */
  void SetMaintainedAll(int64_t maintained_stream) {
    stream_ids_to_bits_[maintained_stream] = ToBit(static_cast<int64_t>(stream_ids_to_bits_.size())) - 1U;
  }

  /**
   * 在观察流`maintained_stream`上，`stream`空出这块资源
   * @param maintained_stream
   * @param stream
   */
  void Clear(int64_t maintained_stream, int64_t stream) {
    stream_ids_to_bits_[maintained_stream] &= ~ToBit(stream);
  }
  void MergeClearedFrom(int64_t src_stream, int64_t dst_stream) {
    stream_ids_to_bits_[dst_stream] &= stream_ids_to_bits_[src_stream];
  }

  /**
   * 在观察流`maintained_stream`上，`stream`是否置位了这块资源
   * @param maintained_stream
   * @param stream
   * @return
   */
  bool IsSet(int64_t maintained_stream, int64_t stream) const {
    return static_cast<bool>(stream_ids_to_bits_[maintained_stream] & ToBit(stream));
  }

  /**
   * 在观察流`maintained_stream`上，是否存在流置位了这块资源
   * @param maintained_stream
   * @return
   */
  bool IsAnySet(int64_t maintained_stream) {
    return stream_ids_to_bits_[maintained_stream] != 0U;
  }

 private:
  static inline uint64_t ToBit(int64_t stream_id) {
    return static_cast<uint64_t>(1) << static_cast<uint64_t>(stream_id);
  }

 private:
  // index: maintained stream id
  ge::SmallVector<uint64_t, kDefaultMaxStreamNum> stream_ids_to_bits_;
};
}  // namespace memory
}  // namespace gert
#endif  // METADEF_CXX_MIF_H_
