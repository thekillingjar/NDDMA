/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GRAPH_CACHE_POLICY_CACHE_DESC_H
#define GRAPH_CACHE_POLICY_CACHE_DESC_H

#include <memory>
#include "graph/utils/hash_utils.h"
namespace ge {
class CacheDesc;
using CacheDescPtr = std::shared_ptr<const CacheDesc>;
class CacheDesc {
 public:
  CacheDesc() = default;
  virtual ~CacheDesc() = default;
  virtual bool IsEqual(const CacheDescPtr &other) const = 0;
  virtual bool IsMatch(const CacheDescPtr &other) const = 0;
  virtual CacheHashKey GetCacheDescHash() const = 0;
};
}  // namespace ge
#endif
