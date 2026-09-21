/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_LOWERING_BUFFER_POOL_H_
#define AIR_CXX_RUNTIME_V2_LOWERING_BUFFER_POOL_H_
#include <unordered_map>
#include <string>
#include <vector>
#include <memory>

namespace gert {
namespace bg {
class BufferPool {
 public:
  using BufId = size_t;
  BufId AddStr(const char *data);
  BufId AddBuf(const uint8_t *data, const size_t len);
  std::unique_ptr<uint8_t[]> Serialize(size_t &total_size) const;
  std::unique_ptr<uint8_t[]> Serialize() const;
  size_t GetSize() const;

  // very slow, only use in UT
  const char *GetBufById(const BufId id) const;

 private:
  BufId AddBuf(std::string &&str);
  BufId AddLargeBuf(std::string &&str);

 private:
  std::unordered_map<std::string, BufId> bufs_to_id_;
  std::vector<std::pair<std::string, BufId>> large_bufs_to_id_; // large buf size, not do hash
  uint64_t id_generator_{0U};
};
}  // namespace bg
}  // namespace gert
#endif  // AIR_CXX_RUNTIME_V2_LOWERING_BUFFER_POOL_H_
