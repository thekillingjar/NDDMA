/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_INC_COMMON_LARGE_BITMAP_H_
#define AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_INC_COMMON_LARGE_BITMAP_H_

#include <list>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "common/fe_log.h"

namespace fe {
class LargeBitMap {
 public:
  explicit LargeBitMap(size_t size);

  ~LargeBitMap() = default;

  bool operator==(const LargeBitMap &another_bm) const;

  bool operator!=(const LargeBitMap &another_bm) const;

  // set all vector to specific value
  void SetValues(uint64_t value);

  // Get the value on position index
  bool GetBit(size_t index) const;

  // Set the value on position index to 1
  void SetBit(size_t index);

  // Combine two bitmap with rule:
  // If one bit of either one of the two bitmaps is 1,
  // the result of final bitmap is 1.
  void Or(const LargeBitMap &another_bm);
 private:
  // Number of element in vector bits
  size_t size_;

  std::vector<uint64_t> bits_;
};
}
#endif //AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_INC_COMMON_LARGE_BITMAP_H_
