/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef METADEF_CXX_TESTS_DEPENDS_CACHE_DESC_STUB_RUNTIME_CACHE_DESC_H
#define METADEF_CXX_TESTS_DEPENDS_CACHE_DESC_STUB_RUNTIME_CACHE_DESC_H
#include <vector>
#include "graph/cache_policy/cache_desc.h"
#include "exe_graph/runtime/shape.h"

namespace ge {
class RuntimeCacheDesc : public CacheDesc {
 public:
  const std::vector<gert::Shape> &GetShapes() const {
    return shapes_;
  }
  void SetShapes(const std::vector<gert::Shape> &shapes) {
    shapes_ = shapes;
  }

  bool IsEqual(const CacheDescPtr &other) const override;
  bool IsMatch(const CacheDescPtr &other) const override;
  CacheHashKey GetCacheDescHash() const override;

 private:
  bool operator==(const RuntimeCacheDesc &rht) const;

 private:
  std::vector<gert::Shape> shapes_;
};
}  // namespace ge
#endif  // METADEF_CXX_TESTS_DEPENDS_CACHE_DESC_STUB_RUNTIME_CACHE_DESC_H
